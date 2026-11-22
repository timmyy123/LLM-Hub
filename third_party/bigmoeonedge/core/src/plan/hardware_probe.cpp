// The machine half of the planner's input. See bmoe/probe.h for the boundary this file is on
// the wrong side of, deliberately: platform conditionals belong here and nowhere above.

#include "bmoe/probe.h"

#include "../io/platform_io.h"
#include "../engine/llama_log.h"
#include "memory_topology.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

namespace bmoe {

namespace {

// What a reclaim costs here. The question the dense policy actually asks is not "which OS is this"
// but "if the kernel wants this memory back, what happens to us", and the answer is read from the
// kernel's own accounting rather than from a platform name: whether this process has a share of its
// own that it is ended for exceeding, whether anonymous memory is compressed first, and whether a
// swap exists. A desktop Linux with zram configured is Compress, and gets the phone's policy for
// the phone's reason; a phone whose kernel keeps a per-process share is Kill whoever made it.
Overflow probe_anon_overflow() {
    pio::ReclaimFacts f;
    if (!pio::reclaim_facts(&f)) return Overflow::Unknown;
    if (f.own_share_left > 0) return Overflow::Kill;
    if (f.compresses) return Overflow::Compress;
    if (f.swaps) return Overflow::Swap;
    // Nothing to reclaim into: anonymous memory cannot be evicted, and pressure ends in a kill.
    return Overflow::Kill;
}

// The core classes this machine has, fastest first, as counts. A barrier waits for its slowest
// participant, so a thread rule that reads a core COUNT on a heterogeneous processor is reading the
// wrong number: on one twenty-core machine the tenth thread is the last fast core and the eleventh
// costs throughput. Where the classes cannot be read the vector stays empty, which every rule reads
// as "this machine did not say" rather than as "every core is alike".
#if defined(__linux__)
// The CPUs this process may be scheduled on, by number; empty where the mask cannot be read.
std::vector<uint32_t> affinity_cpus() {
    std::vector<uint32_t> out;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return out;
    for (uint32_t cpu = 0; cpu < (uint32_t) CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &set)) out.push_back(cpu);
    return out;
}
#endif

// The cores this process can keep busy: the CPUs it may be scheduled on, and no more than a
// control group's quota lets it run at once. Either one leaves every core of the machine visible,
// so the count of the machine is not the count of the process: a container given 32 CPUs of 128
// sweeps to the wrong thread count and measures its rates in a burst it cannot sustain.
uint32_t usable_cores() {
    uint32_t n = std::thread::hardware_concurrency();
#if defined(__linux__)
    const size_t allowed = affinity_cpus().size();
    if (allowed > 0 && (n == 0 || allowed < n)) n = (uint32_t) allowed;
#endif
    const uint32_t quota = pio::cgroup_cpu_limit();
    return quota > 0 && (n == 0 || quota < n) ? quota : n;
}

void probe_core_classes(HardwareProfile & h) {
#if defined(__linux__)
    // Group the CPUs by the maximum frequency the kernel reports for each. It is a fact about this
    // machine, published per CPU, and it needs no table of processor names to interpret. The CPUs
    // are the ones this process may run on, which a quota does not narrow: it bounds how many run
    // at once, not which.
    std::vector<uint32_t> cpus = affinity_cpus();
    if (cpus.empty())
        for (uint32_t cpu = 0; cpu < std::thread::hardware_concurrency(); ++cpu)
            cpus.push_back(cpu);
    std::vector<uint64_t> khz;
    for (uint32_t cpu : cpus) {
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_max_freq", cpu);
        FILE * f = std::fopen(path, "re");
        if (!f) return; // partial information about a heterogeneous machine is worse than none
        unsigned long long v = 0;
        const int n = std::fscanf(f, "%llu", &v);
        std::fclose(f);
        if (n != 1 || v == 0) return;
        khz.push_back((uint64_t) v);
    }
    if (khz.empty()) return;
    std::sort(khz.begin(), khz.end(), std::greater<uint64_t>());
    for (size_t i = 0; i < khz.size();) {
        size_t j = i;
        while (j < khz.size() && khz[j] == khz[i])
            ++j;
        h.core_classes.push_back((uint32_t) (j - i));
        i = j;
    }
#elif defined(__APPLE__)
    // The kernel publishes the classes themselves, fastest first, each with its CPU count.
    uint32_t levels = 0;
    size_t len = sizeof(levels);
    if (sysctlbyname("hw.nperflevels", &levels, &len, nullptr, 0) != 0 || levels == 0) return;
    std::vector<uint32_t> classes;
    for (uint32_t i = 0; i < levels; ++i) {
        char name[64];
        std::snprintf(name, sizeof(name), "hw.perflevel%u.logicalcpu", i);
        uint32_t n = 0;
        len = sizeof(n);
        if (sysctlbyname(name, &n, &len, nullptr, 0) != 0 || n == 0) return; // all the classes or none
        classes.push_back(n);
    }
    h.core_classes = classes;
#else
    // Nothing here reports classes for free. A machine whose cores are alike is still describable,
    // but claiming that without evidence is exactly the guess this design refuses to make.
    (void) h;
#endif
}

void probe_devices(HardwareProfile & h) {
    // Nothing is registered before llama_backend_init(); an empty list here simply means the caller
    // probed early, and every rule that reads devices treats it as "no device-local memory".
    const size_t n_reg = ggml_backend_reg_count();
    for (size_t i = 0; i < n_reg; ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg) continue;
        const size_t n_dev = ggml_backend_reg_dev_count(reg);
        for (size_t j = 0; j < n_dev; ++j) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
            if (!dev) continue;
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(dev, &props);
            // An accelerator in ggml's sense (BLAS, AMX, ...) is not somewhere work can be placed:
            // it is a library the host's own graphs call into for some operations, registered as a
            // device because that is how a backend is made known. It computes on the host's memory
            // with the host's cores, holds no weights of its own, and llama.cpp never places a layer
            // on it. The device type says so, whatever the library is called; listed, it was probed
            // as if it were a device and reported at 0 GiB/s on every plan (#290). A GPU or NPU
            // backend (Hexagon among them) reports GPU or IGPU and is kept.
            if (props.type == GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;

            ComputeDevice d;
            d.name = props.name ? props.name : "";
            d.description = props.description ? props.description : "";
            d.physical_id = props.device_id ? props.device_id : "";
            d.memory_free = props.memory_free;
            d.memory_total = props.memory_total;
            d.is_cpu = props.type == GGML_BACKEND_DEVICE_TYPE_CPU;
            // The device TYPE may only ever narrow this to "shared", never to "its own". The CPU's
            // memory is the host's by definition, and a device that calls itself integrated is
            // claiming shared memory - a claim worth believing, because believing it errs safe.
            // `TYPE_GPU` is not the opposite claim: Metal reports it on unified-memory hardware, so
            // taking it as evidence of separate memory is how one physical pool gets counted twice.
            // Native topology can settle it below; otherwise it stays Unknown - read as shared -
            // until probe_device_costs allocates on the device and watches what the host loses.
            if (d.is_cpu || props.type == GGML_BACKEND_DEVICE_TYPE_IGPU) d.shares_host_memory = Tri::Yes;
            // A backend that reports TYPE_GPU for every device (Metal on unified memory, OpenCL on an
            // integrated GPU) is asked through its own API, which the driver answers as a fixed
            // property; the allocation delta in probe_device_costs stays for whatever neither tells.
            if (!d.is_cpu)
                d.shares_host_memory = detail::refine_memory_sharing(
                    d.shares_host_memory, detail::native_memory_sharing(ggml_backend_reg_name(reg), props.description));
            // Whether this device will execute over memory the host owns. Asking the backend for a
            // host buffer type is the question; a device that answers is one we can read flash into
            // and it can compute out of, which is exactly the pair of properties a streamed expert
            // needs. Host memory has it by definition, and a device with memory of its own may still
            // offer it - which is the difference between an accelerator that can serve streamed
            // experts and one that can only be handed a copy it repacks.
            d.host_buffer =
                d.reads_host_memory() || ggml_backend_dev_host_buffer_type(dev) != nullptr ? Tri::Yes : Tri::No;
            // Only a host address can be repointed at bytes we read ourselves, which is the
            // condition the expert streamer exists under. A device-local buffer's pointer is not one.
            d.rebindable = is_yes(d.host_buffer);
            // Two more capabilities the device declares for free. Whether it can copy while it
            // computes is a plain fact. Whether it can wrap memory we already own - the one thing a
            // streamed expert would need - is NOT, and the mapping below is asymmetric on purpose.
            //
            // `caps.buffer_from_host_ptr` is a declaration, and a false one does not establish
            // absence. In the pinned submodule the Vulkan, OpenCL and SYCL devices each report it
            // false while wiring a complete implementation into their device interface: Vulkan's
            // imports host memory through an extension whose success is per-device and per-call, so
            // there is nothing it could honestly promise in a blanket capability, and it promises
            // nothing. A rule reading No here would conclude "closed" about the backends that run on
            // the machines this project targets.
            //
            // Nor can we settle it by trying. Unlike its neighbour `get_host_buffer_type`, which
            // null-checks the interface pointer and returns nothing, `ggml_backend_dev_buffer_from_
            // host_ptr` calls straight through - so an attempt on a device that genuinely lacks it
            // dereferences null instead of declining. Until an instrument exists that can separate
            // the two, a yes is a fact and everything else is Unknown: the same shape as the
            // mapping-serialisation probe, and for the same reason.
            d.host_ptr_buffers = props.caps.buffer_from_host_ptr ? Tri::Yes : Tri::Unknown;
            d.async_copies = props.caps.async ? Tri::Yes : Tri::No;
            // Whether this device's buffers come out of the total the process locks against. The
            // CPU's are ordinary host memory and do not. For the rest it is a property of how the
            // platform backs a device allocation, which no call here reports: Unknown, which the
            // ledger charges like a yes - the reading that cannot overcommit, and the one observed
            // where it was measured (a device prefill took the system's wired count to 14.4 GB of 16).
            if (d.is_cpu) d.buffers_locked = Tri::No;
            h.devices.push_back(std::move(d));
        }
    }
}

} // namespace

// How many backends this binary carries, and how many it found beside itself. Recorded rather than
// recomputed, because the difference is only visible at the moment of loading — and it is the
// difference between two answers a plan must never conflate: "this machine has no accelerator" and
// "this build never went to look".
uint32_t g_backends_linked = 0;
uint32_t g_backends_loaded = 0;
bool g_backends_looked = false;

namespace {

// How ggml names a backend module (ggml-backend-reg.cpp: backend_filename_prefix/extension).
#if defined(_WIN32)
constexpr const char * kModulePrefix = "ggml-";
constexpr const char * kModuleExtension = ".dll";
#else
constexpr const char * kModulePrefix = "libggml-";
constexpr const char * kModuleExtension = ".so";
#endif

// "cuda" for libggml-cuda.so, "cpu" for libggml-cpu-haswell.so; "" for a file that is no module.
std::string module_backend(const std::string & file) {
    const std::string prefix = kModulePrefix, ext = kModuleExtension;
    if (file.size() <= prefix.size() + ext.size() || file.compare(0, prefix.size(), prefix) != 0 ||
        file.compare(file.size() - ext.size(), ext.size(), ext) != 0)
        return "";
    const std::string rest = file.substr(prefix.size(), file.size() - prefix.size() - ext.size());
    const size_t dash = rest.find('-');
    return dash == std::string::npos ? rest : rest.substr(0, dash);
}

std::vector<std::string> files_in(const std::string & dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(std::filesystem::u8path(dir), ec), end; !ec && it != end;
         it.increment(ec)) {
        std::error_code fec;
        if (it->is_regular_file(fec)) out.push_back(it->path().filename().u8string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

std::vector<std::string> extra_backend_modules(const std::string & exe_dir,
                                               const std::vector<std::string> & extra_dirs) {
    std::set<std::string> taken, beside_exe;
    for (const std::string & f : files_in(exe_dir)) {
        beside_exe.insert(f);
        const std::string b = module_backend(f);
        if (!b.empty()) taken.insert(b);
    }
    std::vector<std::string> out;
    for (const std::string & dir : extra_dirs) {
        if (dir.empty()) continue;
        // Per backend, the module named after it alone when there is one, else the first variant.
        std::map<std::string, std::string> pick;
        for (const std::string & f : files_in(dir)) {
            const std::string b = module_backend(f);
            if (b.empty() || beside_exe.count(f) || taken.count(b)) continue;
            const bool plain = f == kModulePrefix + b + kModuleExtension;
            if (!pick.count(b) || plain) pick[b] = f;
        }
        for (const auto & [b, f] : pick) {
            taken.insert(b);
            out.push_back((std::filesystem::u8path(dir) / std::filesystem::u8path(f)).u8string());
        }
    }
    return out;
}

void register_backends(const std::vector<std::string> & extra_dirs) {
    // Once: llama_backend_init() loads nothing when the registry is already filled, and a second
    // pass here would count the modules the first one loaded as linked in.
    if (g_backends_looked) return;
    // The first thing every front-end and every probe child does with llama.cpp, so the log bridge
    // goes in here: loading the backends already logs, and so does every model load after it.
    detail::install_llama_log();
    // What the registry holds before anything is loaded is what came linked in (the CPU backend
    // in a build that links it, nothing in one that loads it). Then the modules: ggml looks for
    // libraries named after each backend, and left to itself it looks in the working directory
    // too, which is wherever the process happened to be started from. Only the directory the
    // executable is in is a place a release put files; a module it did not ship is not loaded.
    g_backends_linked = (uint32_t) ggml_backend_reg_count();
    const std::string self = pio::self_path();
    if (!self.empty()) {
        const size_t slash = self.find_last_of("/\\");
        const std::string dir = slash == std::string::npos ? std::string(".") : self.substr(0, slash);
        ggml_backend_load_all_from_path(dir.c_str());
        // Then the folders the front-end installed backends into, one module per backend the
        // executable's folder did not already supply, so no device is registered twice.
        for (const std::string & m : extra_backend_modules(dir, extra_dirs))
            ggml_backend_load(m.c_str());
    } else {
        ggml_backend_load_all();
    }
    const uint32_t after = (uint32_t) ggml_backend_reg_count();
    g_backends_loaded = after > g_backends_linked ? after - g_backends_linked : 0;
    g_backends_looked = true;
}

BackendInventory backend_inventory() {
    BackendInventory inv;
    inv.linked = g_backends_linked;
    inv.loaded_at_runtime = g_backends_loaded;
    inv.looked = g_backends_looked;
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (!reg) continue;
        inv.names.push_back(ggml_backend_reg_name(reg));
        // Read from what a backend offers, not from its name. A backend whose devices are added by
        // naming a server is remote: it offers nothing local whatever this machine has. The host's
        // own backend and a library the host calls into have devices of their own kind. Anything
        // else could offer a device here, and offering none on this machine is then the answer.
        if (ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_add_server")) continue;
        const size_t n = ggml_backend_reg_dev_count(reg);
        bool host_side = n > 0;
        for (size_t j = 0; j < n; ++j) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
            const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
            if (t != GGML_BACKEND_DEVICE_TYPE_CPU && t != GGML_BACKEND_DEVICE_TYPE_ACCEL) host_side = false;
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                const char * d = ggml_backend_dev_description(dev);
                inv.graphics.push_back({ggml_backend_reg_name(reg), d ? d : ""});
            }
        }
        if (!host_side) ++inv.placeable;
    }
    return inv;
}

HardwareProfile probe_hardware(const char * model_path) {
    HardwareProfile h;

    h.residency_budget = pio::mem_available_bytes();
    h.memory_total = pio::mem_total_bytes();
    h.anon_overflow = probe_anon_overflow();
    h.reclaim_exempt_max = pio::pinned_max_bytes();
    // A denied query is not a zero grant or a measured unlimited store. Carry its uncertainty
    // beside any independently known finite bound so the plan can make a bounded runtime attempt.
    const pio::LockLimit lockable = pio::lockable_limit();
    const pio::LockLimit in_place = pio::lock_in_place_limit();
    h.lockable_bytes = lockable.bytes;
    h.lockable_bytes_known = lockable.known;
    h.lock_in_place_bytes = in_place.bytes;
    h.lock_in_place_bytes_known = in_place.known;
    h.discardable_memory = pio::vm_discard_supported();
    h.n_cores = usable_cores();
    probe_core_classes(h);

    const BackendInventory inv = backend_inventory();
    h.backends_linked = inv.linked;
    h.backends_loaded = inv.loaded_at_runtime;
    h.backends_looked = inv.looked;
    h.backends_placeable = inv.placeable;

    probe_devices(h);

    if (model_path && *model_path) {
        // Whether uncached reads are actually in effect is a property of this path, not of the
        // platform: the same device answers yes on one filesystem and declines on another. Ask the
        // open what it got rather than reconstructing it from what was requested.
        bool effective = false;
        pio::fd_t fd = pio::open_read(model_path, /*direct=*/true, &effective);
        if (pio::fd_ok(fd)) {
            h.storage.direct_ok = effective ? Tri::Yes : Tri::No;
            pio::close_fd(fd);
        }
    }

    // Left Unknown on purpose: both cost real I/O to answer, and a plan that says "unprobed" is
    // worth more than one that assumes. See bmoe/probe.h.
    h.storage.mapping_serialises_reads = Tri::Unknown;

    h.label = std::to_string(h.n_cores) + " cores, " + std::to_string((unsigned long long) (h.residency_budget >> 20)) +
              " MiB available of " + std::to_string((unsigned long long) (h.memory_total >> 20)) + " MiB";
    return h;
}

void probe_device_support(HardwareProfile & hw, const ModelProfile & model) {
    if (model.expert_type_id < 0 || model.n_expert == 0) return; // nothing to ask about

    // Build the real operation in a context that allocates no tensor data: a MUL_MAT_ID over a 3-D
    // expert tensor of this model's own type and shape. Asking about the actual node is the whole
    // point - a table of "supported formats" would go stale the moment a backend gains a kernel,
    // and would have to be written once per vendor.
    ggml_init_params ip{};
    ip.mem_size = 4 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return;

    // One op per expert layout the file carries: a mixed quant is executed in every one of its
    // formats, so a device that lacks a kernel for any of them cannot run the experts as a whole.
    // Asked for each rather than for the commonest, which is what this used to do.
    std::vector<ggml_tensor *> ops;
    const std::vector<int> types =
        model.expert_type_ids.empty() ? std::vector<int>{model.expert_type_id} : model.expert_type_ids;
    for (int id : types) {
        const ggml_type type = (ggml_type) id;
        const int64_t blk = ggml_blck_size(type);
        if (blk <= 0) continue;
        // Shapes only have to be legal and representative; the graph is never run.
        const int64_t n_embd = blk * 8;
        const int64_t n_ff = blk * 8;
        const int64_t n_used = (int64_t) (model.n_expert_used ? model.n_expert_used : 1);

        // Shapes follow ggml_mul_mat_id's own contract: `as` is 3-D with one matrix per expert,
        // `b` is 3-D whose ne[2] is the token count, and `ids` is 2-D [n_expert_used, n_tokens].
        // One token is enough to ask the question, and keeping it at one keeps the assertions
        // trivially satisfied.
        ggml_tensor * w = ggml_new_tensor_3d(ctx, type, n_embd, n_ff, (int64_t) model.n_expert);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
        if (w && x && ids) ops.push_back(ggml_mul_mat_id(ctx, w, x, ids));
    }
    if (!ops.empty() && ops.size() == types.size()) {
        for (ComputeDevice & d : hw.devices) {
            if (!d.is_cpu && !d.probe_here()) continue; // asked in another process
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(d.name.c_str());
            if (!dev) continue;
            bool all = true;
            for (ggml_tensor * op : ops)
                all = all && ggml_backend_dev_supports_op(dev, op);
            d.runs_expert_op = all ? Tri::Yes : Tri::No;

            // A device that executes this layout only out of a buffer type of its own needs the
            // weight converted at load. For a streamed expert that is fatal rather than merely
            // costly: the streamer's whole mechanism is rebinding `data` onto the file's native
            // layout, and a repack is precisely what replaces that layout.
            // Ask about the buffer type this weight would ACTUALLY live in. For a streamed
            // expert that is the host buffer where one is offered, not the device's default:
            // a discrete accelerator's own buffer needs a repack and its host buffer does not,
            // and reading the wrong one of the two is the difference between "this device
            // cannot serve streamed experts" and the truth.
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
            if (!buft) buft = ggml_backend_dev_buffer_type(dev);
            d.needs_repack = (buft && !ggml_backend_buft_is_host(buft)) ? Tri::Yes : Tri::No;
        }
    }
    ggml_free(ctx);
}

} // namespace bmoe
