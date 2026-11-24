#include "platform_io.h"

#include <algorithm>
#include <atomic>
#include <filesystem>

// System headers MUST be included at global scope, never inside the namespace below:
// <cstdlib> etc. do `using ::abs;` and would otherwise be pulled into bmoe::pio, where
// ::abs is not visible (GCC hard-errors; MSVC happened to tolerate it).
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h> // QueryWorkingSetEx: kernel32's K32 export on every supported Windows
#include <malloc.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>
#else
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <dirent.h>
#include <chrono>
#include <string>
#include <ctime>
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0 // absent on some BSDs; the mapping is simply charged as usual there
#endif
#if defined(__linux__)
#include <sys/syscall.h> // statx by number; see direct_alignment
#endif
#if defined(__ANDROID__)
#include <android/api-level.h>       // whether the sandbox admits statx; see direct_alignment
#include <android/hardware_buffer.h> // reclaim-exempt allocation; see pinned_alloc
#endif
#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <mach-o/dyld.h> // this executable's own path; see run_self
#include <mach/mach.h>
#include <mach/vm_purgable.h> // discardable cache spans; see vm_offer
#include <sys/sysctl.h>
#if TARGET_OS_IPHONE
#include <os/proc.h> // this process's own share of memory; see mem_available_bytes
#else
#include <libproc.h> // this process's own regions; see file_mapped_regions
#endif
#endif
#endif

namespace bmoe::pio {

namespace {
// What the platform names as this executable can be a link to it: an installer that puts a link on
// PATH and the files in a folder of their own, a package manager's shim. Everything a release ships
// sits beside the file itself, so the path that counts is the one with every link resolved. Where
// it cannot be resolved the name as given is still the best answer there is.
std::string without_links(const std::string & path) {
    if (path.empty()) return path;
    // UTF-8 in and out, as every path here is: read as the system's own narrow encoding, a path
    // outside it would not resolve, and one inside it would come back in bytes that are not UTF-8.
    std::error_code ec;
    const std::filesystem::path real = std::filesystem::canonical(std::filesystem::u8path(path), ec);
    return ec ? path : real.u8string();
}
} // namespace

#if !defined(_WIN32)
// mincore's vector argument is `unsigned char *` on Linux/Android but `char *` on the BSDs and
// macOS. Name the difference once instead of casting blind at the call site.
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
using mincore_vec_t = char;
#else
using mincore_vec_t = unsigned char;
#endif
#endif

#if defined(_WIN32)

const fd_t fd_invalid = (void *) INVALID_HANDLE_VALUE;
bool fd_ok(fd_t fd) {
    return fd != (void *) INVALID_HANDLE_VALUE;
}

fd_t open_read(const char * path, bool direct, bool * effective_direct) {
    DWORD flags = FILE_ATTRIBUTE_NORMAL | (direct ? FILE_FLAG_NO_BUFFERING : 0);
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
    if (effective_direct) *effective_direct = direct && fd_ok((fd_t) h);
    return (fd_t) h;
}

void close_fd(fd_t fd) {
    if (fd_ok(fd)) CloseHandle((HANDLE) fd);
}

long long pread_at(fd_t fd, void * buf, size_t count, uint64_t off) {
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD) (off & 0xFFFFFFFFull);
    ov.OffsetHigh = (DWORD) (off >> 32);
    // FILE_FLAG_NO_BUFFERING rejects a length that is not a multiple of the sector
    // size, so cap per-call length to a sector-aligned 1 GiB chunk (0x7FFFFFFF is odd).
    DWORD to_read = count > 0x40000000ull ? 0x40000000ul : (DWORD) count;
    DWORD got = 0;
    if (!ReadFile((HANDLE) fd, buf, to_read, &got, &ov)) {
        return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    }
    return (long long) got;
}

uint64_t file_size(fd_t fd) {
    LARGE_INTEGER sz;
    return GetFileSizeEx((HANDLE) fd, &sz) ? (uint64_t) sz.QuadPart : 0;
}

void * alloc_aligned(size_t align, size_t sz) {
    return _aligned_malloc(sz, align);
}
void aligned_free(void * p) {
    if (p) _aligned_free(p);
}

size_t vm_page() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t) si.dwPageSize;
}
void * vm_reserve(size_t sz) {
    return VirtualAlloc(nullptr, sz, MEM_RESERVE, PAGE_READWRITE);
}
bool vm_commit(void * p, size_t sz) {
    return VirtualAlloc(p, sz, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}
bool vm_pin(void * /*p*/, size_t /*sz*/) {
    return false; // VirtualLock is bounded by the working-set quota, which is not ours to raise
}
void vm_evict(void * p, size_t sz) {
    if (sz) VirtualFree(p, sz, MEM_DECOMMIT);
}
void vm_release(void * p, size_t /*sz*/) {
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}
namespace {
// OfferVirtualMemory arrived in Windows 8.1; resolved at run time so the build never depends on it.
using OfferFn = DWORD(WINAPI *)(PVOID, SIZE_T, DWORD);
using ReclaimFn = DWORD(WINAPI *)(PVOID, SIZE_T);
struct OfferApi {
    OfferFn offer = nullptr;
    ReclaimFn reclaim = nullptr;
    OfferApi() {
        if (HMODULE k = GetModuleHandleW(L"kernel32.dll")) {
            offer = (OfferFn) (void *) GetProcAddress(k, "OfferVirtualMemory");
            reclaim = (ReclaimFn) (void *) GetProcAddress(k, "ReclaimVirtualMemory");
        }
    }
};
const OfferApi & offer_api() {
    static const OfferApi api;
    return api;
}
} // namespace
bool vm_discard_supported() {
    // The API exists, but it does not pay here: OfferVirtualMemory takes the range out of the working
    // set and makes it PAGE_NOACCESS at once, kept or not, and the offer and the reclaim each cost
    // microseconds per page (about 1 ms per 1.7 MiB entry, twice that with the decode threads busy),
    // paid on every reuse of an offered entry. Measured on a 15 GB Windows 11 PC with
    // Qwen3.6-35B-A3B Q4_K_M: decode 2.2-2.9 tok/s offered against 3.7-4.8 with an ordinary cache, the
    // same bytes and hits, the whole gap in cache management. Until an offer that keeps its pages
    // mapped is found, the cache here is ordinary memory, as it was before the discardable cache.
    return false;
}
bool vm_commit_discardable(void * p, size_t sz) {
    return vm_commit(p, sz);
}
bool vm_offer(void * p, size_t sz) {
    constexpr DWORD offer_priority_low = 2; // VmOfferPriorityLow: taken before ordinary standby pages
    return sz && vm_discard_supported() && offer_api().offer(p, sz, offer_priority_low) == ERROR_SUCCESS;
}
bool vm_reclaim(void * p, size_t sz) {
    if (!sz || !vm_discard_supported()) return true;
    // ERROR_BUSY: some pages were discarded. The range is committed again either way.
    return offer_api().reclaim(p, sz) == ERROR_SUCCESS;
}
int vm_offered_held(const void * /*p*/, size_t /*sz*/) {
    // An offered range leaves the working set and turns PAGE_NOACCESS the moment it is offered, kept
    // or not, so QueryWorkingSetEx reads every offer as gone. Only ReclaimVirtualMemory tells, and
    // that takes the range back.
    return -1;
}
void vm_drop_file_pages(void * /*p*/, size_t /*sz*/) {
    // File-backed views cannot be decommitted (MEM_DECOMMIT is only valid for VirtualAlloc'd pages),
    // and the host build never mmaps the model for streaming — so there is nothing to drop.
}
void drop_file_cache(const char * /*path*/) {
    // No posix_fadvise; the standby list is the system's to trim.
}
void drop_file_cache_range(const char * /*path*/, uint64_t /*offset*/, uint64_t /*size*/) {
    // As above.
}
void vm_drop_anon_pages(void * /*p*/, size_t /*sz*/) {
    // MEM_DECOMMIT would turn the owner's next write into an access violation, and MEM_RESET is only
    // defined for VirtualAlloc'd regions, which a heap allocation need not be. The host build has no
    // prefill device that moves state away, so keeping the pages costs nothing here.
}

void vm_advise_random(void * /*p*/, size_t /*sz*/) {
    // No readahead to tame on a host build that never streams; the gates do not measure I/O.
}

// The same question mincore answers elsewhere, asked of this process's working set: a page whose
// entry is Valid is resident. Clipped to whole pages exactly as the POSIX version is, so the two
// fractions mean the same thing.
bool vm_resident_sample(const void * p, size_t sz, size_t * sampled, size_t * resident) {
    const size_t page = vm_page();
    uintptr_t a0 = ((uintptr_t) p + page - 1) & ~(uintptr_t) (page - 1);
    uintptr_t a1 = ((uintptr_t) p + sz) & ~(uintptr_t) (page - 1);
    if (a1 <= a0) return true;
    PSAPI_WORKING_SET_EX_INFORMATION info[256];
    for (uintptr_t a = a0; a < a1;) {
        const size_t want = std::min<size_t>((size_t) (a1 - a) / page, sizeof(info) / sizeof(info[0]));
        for (size_t i = 0; i < want; ++i)
            info[i].VirtualAddress = (PVOID) (a + i * page);
        if (!QueryWorkingSetEx(GetCurrentProcess(), info, (DWORD) (want * sizeof(info[0])))) return false;
        for (size_t i = 0; i < want; ++i)
            *resident += (size_t) info[i].VirtualAttributes.Valid;
        *sampled += want;
        a += want * page;
    }
    return true;
}

bool file_mapped_regions(const char * /*basename*/, std::vector<MappedRegion> & /*out*/) {
    return false; // no /proc/self/maps; the host gates do not exercise dense residency
}

uint64_t mem_available_bytes() {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullAvailPhys : 0;
}

uint64_t mem_total_bytes() {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
}

// No per-process count of HARD faults is published here: PageFaultCount adds the soft ones, which
// would read a cheap remap as a flash stall. So this stays unmeasured (0), and the dense residency
// sensor above is the instrument that sees a refaulting dense set on this platform.
uint64_t major_faults() {
    return 0;
}
bool major_faults_counted() {
    return false;
}
double process_cpu_seconds() {
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0.0;
    auto secs = [](const FILETIME & f) {
        return (double) (((uint64_t) f.dwHighDateTime << 32) | f.dwLowDateTime) * 1e-7; // 100 ns ticks
    };
    return secs(kernel) + secs(user);
}

size_t fault_bytes() {
    return vm_page();
}

bool process_memory(ProcessMemory * /*out*/) {
    return false;
}

bool device_memory(DeviceMemory * out) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return false;
    out->available_bytes = (uint64_t) ms.ullAvailPhys;
    out->free_bytes = (uint64_t) ms.ullAvailPhys; // no separate "free vs reclaimable" here
    out->swap_free_bytes = (uint64_t) ms.ullAvailPageFile;
    return true;
}

// The commit limit is RAM plus the pagefiles, so a limit above RAM is a pagefile that exists. What
// the memory manager compresses first is not published to a process, so it is not claimed.
bool reclaim_facts(ReclaimFacts * out) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return false;
    out->swaps = ms.ullTotalPageFile > ms.ullTotalPhys;
    return true;
}

uint64_t cgroup_memory_left(const char *, const char *) {
    return 0; // job objects, not control groups, cap a process here; none is read yet
}

uint32_t cgroup_cpu_limit(const char *, const char *) {
    return 0;
}

#else

const fd_t fd_invalid = -1;
bool fd_ok(fd_t fd) {
    return fd >= 0;
}

fd_t open_read(const char * path, bool direct, bool * effective_direct) {
#if defined(__APPLE__)
    // No O_DIRECT here (the shim above leaves it 0), so a direct request is an ordinary open plus
    // F_NOCACHE: ask the kernel not to keep this descriptor's pages. That is the whole of Apple's
    // uncached mode — a caching hint on the fd, not an I/O mode — so a refusal is a downgrade to
    // buffered for the caller to report, never a reason to fail a perfectly good descriptor.
    const fd_t fd = open(path, O_RDONLY | O_CLOEXEC);
    const bool ok = fd_ok(fd) && direct && fcntl(fd, F_NOCACHE, 1) == 0;
    if (effective_direct) *effective_direct = ok;
    return fd;
#else
    const fd_t fd = open(path, O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
    if (effective_direct) *effective_direct = direct && fd_ok(fd);
    return fd;
#endif
}

void close_fd(fd_t fd) {
    if (fd_ok(fd)) close(fd);
}

long long pread_at(fd_t fd, void * buf, size_t count, uint64_t off) {
    return (long long) pread(fd, buf, count, (off_t) off);
}

uint64_t file_size(fd_t fd) {
    // fstat rather than a seek pair: every consumer reads with pread, so the fd's file position is
    // never used — and mutating shared fd state to answer a question about the file is a trap waiting
    // for the first caller that does rely on the position.
    struct stat st;
    return fstat(fd, &st) == 0 && st.st_size > 0 ? (uint64_t) st.st_size : 0;
}

void * alloc_aligned(size_t align, size_t sz) {
    void * p = nullptr;
    return posix_memalign(&p, align, sz) == 0 ? p : nullptr;
}
void aligned_free(void * p) {
    free(p);
}

size_t vm_page() {
    // sysconf does not fail for this name in practice; if it ever did, the smallest page any target
    // has is the one value that cannot make a page-clipped range spill outside its allocation.
    constexpr size_t smallest_page = 4096;
    long ps = sysconf(_SC_PAGESIZE);
    return ps > 0 ? (size_t) ps : smallest_page;
}
void * vm_reserve(size_t sz) {
    // MAP_NORESERVE makes the "address-only" contract true rather than merely true-by-default: these
    // spans are reserved at full expert-set size but only ever committed for resident slices, so the
    // untouched remainder must not be charged against the commit limit (which is what strict
    // overcommit would do, refusing a reservation the cache never intends to fill).
    void * p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}
bool vm_commit(void * /*p*/, size_t /*sz*/) {
    return true; // POSIX commits on first touch
}
namespace {
std::atomic<bool> g_pinned_any{false};
}
bool vm_pin(void * p, size_t sz) {
    if (!sz) return true;
    if (mlock(p, sz) != 0) return false;
    g_pinned_any.store(true, std::memory_order_relaxed);
    return true;
}
void vm_evict(void * p, size_t sz) {
    if (!sz) return;
#if defined(__APPLE__)
    // Darwin's MADV_DONTNEED is advice and frees nothing: the pages of an evicted expert stay
    // dirty and anonymous, so the process grows towards the whole expert set whatever the cache
    // budget says, and the kernel answers by compressing it - the live cache included. Every hit
    // then pays a decompression that no fault counter shows. Measured on a 16 GB machine with a
    // 9 GB budget: 8.4 GB of this process in the compressor and decode at a third of its rate.
    // Mapping fresh zero-fill pages over the span is the release that cannot be declined; the
    // address stays valid, which is the contract.
    if (mmap(p, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) !=
        MAP_FAILED)
        return;
    madvise(p, sz, MADV_FREE); // lazily, if the remap was refused
#else
    // A locked range refuses MADV_DONTNEED, so a pinned slice is unlocked before it is dropped.
    if (g_pinned_any.load(std::memory_order_relaxed)) munlock(p, sz);
    madvise(p, sz, MADV_DONTNEED);
#endif
}
void vm_release(void * p, size_t sz) {
    if (p) munmap(p, sz);
}
#if defined(__APPLE__)
bool vm_discard_supported() {
    return true;
}
bool vm_commit_discardable(void * p, size_t sz) {
    // A purgeable object at exactly this span, replacing the zero-fill mapping vm_evict left there.
    // Measured on a 16 GB machine under 9 GB of pressure: every volatile object emptied, nothing
    // swapped; a state change costs ~0.5 us where madvise(MADV_FREE_REUSE) costs ~25 us per 1.7 MB.
    vm_address_t a = (vm_address_t) p;
    return vm_allocate(mach_task_self(), &a, sz, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE | VM_FLAGS_PURGABLE) ==
               KERN_SUCCESS &&
           a == (vm_address_t) p;
}
bool vm_offer(void * p, size_t sz) {
    if (!sz) return false;
    int state = VM_PURGABLE_VOLATILE;
    return vm_purgable_control(mach_task_self(), (vm_address_t) p, VM_PURGABLE_SET_STATE, &state) == KERN_SUCCESS;
}
bool vm_reclaim(void * p, size_t sz) {
    if (!sz) return true;
    int state = VM_PURGABLE_NONVOLATILE;
    if (vm_purgable_control(mach_task_self(), (vm_address_t) p, VM_PURGABLE_SET_STATE, &state) != KERN_SUCCESS)
        return false; // not an object we offered: the caller re-reads, which is always safe
    // The previous state comes back in `state`; an emptied object is zero-fill from here on.
    return (state & VM_PURGABLE_STATE_MASK) != VM_PURGABLE_EMPTY;
}
#elif defined(__linux__) && defined(MADV_FREE) // Linux and Android: "a write cancels the free" is Linux semantics
bool vm_discard_supported() {
    return true;
}
bool vm_commit_discardable(void * p, size_t sz) {
    return vm_commit(p, sz);
}
bool vm_offer(void * p, size_t sz) {
    // Refused (EINVAL) on kernels before 4.5 and on locked ranges: the span stays ordinary memory.
    return sz && madvise(p, sz, MADV_FREE) == 0;
}
bool vm_reclaim(void * p, size_t sz) {
    if (!sz) return true;
    // A write to a page cancels its pending free; a page already freed comes back as a fresh zero
    // page whatever is written to it. So: write one word back to itself, THEN look at a different
    // word. Once written the page can no longer be dropped, so a page that reads non-zero away from
    // the written word is the one that was offered. The order is what makes it race-free - checking
    // first would let the kernel drop the page between the check and the use.
    // A page whose every other word is zero reads as dropped although it may not be: that costs a
    // read, never a wrong answer, and the scan stops at the first non-zero word on any real weight.
    const size_t page = vm_page(), words = page / sizeof(uint64_t);
    bool intact = true;
    for (char * pg = (char *) p; pg < (char *) p + sz; pg += page) {
        volatile uint64_t * w = (volatile uint64_t *) pg;
        w[0] = w[0];
        if (!intact) continue; // the caller refills the whole span; the writes above still cancel frees
        size_t i = 1;
        while (i < words && w[i] == 0)
            ++i;
        intact = i < words;
    }
    return intact;
}
#else
bool vm_discard_supported() {
    return false;
}
bool vm_commit_discardable(void * p, size_t sz) {
    return vm_commit(p, sz);
}
bool vm_offer(void *, size_t) {
    return false;
}
bool vm_reclaim(void *, size_t) {
    return true;
}
#endif
void vm_drop_file_pages(void * p, size_t sz) {
    // MADV_DONTNEED on the model's clean, read-only MAP_PRIVATE mapping drops the resident pages; the
    // next access refaults them from the file. The tensor was rebound onto its anon copy, so nothing
    // touches this range again — the drop just reclaims the double residency, it does not lose data.
    if (sz) madvise(p, sz, MADV_DONTNEED);
}
void drop_file_cache(const char * path) {
#if defined(__APPLE__)
    (void) path; // no posix_fadvise on Darwin
#else
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); // advice only: a failure keeps the pages
    ::close(fd);
#endif
}
void drop_file_cache_range(const char * path, uint64_t offset, uint64_t size) {
#if defined(__APPLE__)
    (void) path, (void) offset, (void) size; // no posix_fadvise on Darwin
#else
    if (size == 0) return;
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    posix_fadvise(fd, (off_t) offset, (off_t) size, POSIX_FADV_DONTNEED); // advice only, as above
    ::close(fd);
#endif
}
void vm_drop_anon_pages(void * p, size_t sz) {
    // On private anonymous memory MADV_DONTNEED frees the pages and leaves the range mapped: a later
    // read sees zeros, a later write faults in a fresh page. Advice only; a failure keeps the pages.
    if (sz) madvise(p, sz, MADV_DONTNEED);
}

void vm_advise_random(void * p, size_t sz) {
    // MADV_RANDOM disables readahead for the range: each fault maps exactly the page that faulted.
    // Advice only, so a failure changes nothing but the readahead and is not worth reporting.
    if (sz) madvise(p, sz, MADV_RANDOM);
}

bool vm_resident_sample(const void * p, size_t sz, size_t * sampled, size_t * resident) {
    if (!p || sz == 0) return true; // an empty range is measured, and holds nothing
    const size_t page = vm_page();
    // Clip to the pages FULLY inside the range, matching how the eviction path releases them: an
    // edge page shared with a neighbouring slice belongs to that neighbour, not to this sample.
    uintptr_t a0 = ((uintptr_t) p + page - 1) & ~(uintptr_t) (page - 1);
    uintptr_t a1 = ((uintptr_t) p + sz) & ~(uintptr_t) (page - 1);
    if (a1 <= a0) return true;
    unsigned char vec[512]; // one byte per page: 512 pages (2 MiB at 4 KiB pages) per syscall
    for (uintptr_t a = a0; a < a1;) {
        const size_t want = std::min<size_t>((size_t) (a1 - a) / page, sizeof(vec));
        if (mincore((void *) a, want * page, (mincore_vec_t *) vec) != 0) return false;
        for (size_t i = 0; i < want; ++i)
            *resident += (size_t) (vec[i] & 1u); // bit 0 = the page is resident
        *sampled += want;
        a += want * page;
    }
    return true;
}
int vm_offered_held(const void * p, size_t sz) {
    // An offered page stays mapped until the kernel takes it, so residency is the answer. One page
    // tells for the span: the kernel drops an offered span whole on every platform that reports it.
    size_t sampled = 0, resident = 0;
    if (!vm_resident_sample(p, std::min(sz, vm_page()), &sampled, &resident) || sampled == 0) return -1;
    return resident != 0 ? 1 : 0;
}

#if defined(__APPLE__)
namespace {
// Darwin's counterpart of /proc/meminfo is the host's own page accounting.
bool host_vm(vm_statistics64_data_t * vm) {
    static const mach_port_t host = mach_host_self(); // a send right; taken once, not per call
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    return host_statistics64(host, HOST_VM_INFO64, (host_info64_t) vm, &n) == KERN_SUCCESS;
}
// Pages nobody is using, as the kernel counts them: free_count includes the speculative read-ahead
// pages, which are file-backed and counted again under external.
uint64_t host_free_pages(const vm_statistics64_data_t & vm) {
    return vm.free_count > vm.speculative_count ? vm.free_count - vm.speculative_count : 0;
}
// The same definition as MemAvailable, built from the parts Darwin publishes: what can be handed
// out without compressing anything — free pages, file-backed pages (dropped, not compressed) and
// purgeable ones. It inherits MemAvailable's blind spot on purpose, so the two stay comparable: a
// mapped model's resident pages are file-backed and count as available here too.
uint64_t host_available_pages(const vm_statistics64_data_t & vm) {
    return host_free_pages(vm) + vm.external_page_count + vm.purgeable_count;
}
} // namespace
#endif

uint64_t mem_total_bytes() {
    if (FILE * f = std::fopen("/proc/meminfo", "re")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) {
                std::fclose(f);
                return (uint64_t) kb * 1024ull;
            }
        }
        std::fclose(f);
    }
#if defined(_SC_PHYS_PAGES)
    const long total = sysconf(_SC_PHYS_PAGES);
    const long psz = sysconf(_SC_PAGESIZE);
    if (total > 0 && psz > 0) return (uint64_t) total * (uint64_t) psz;
#endif
    return 0;
}

uint64_t mem_available_bytes() {
#if defined(__APPLE__)
#if TARGET_OS_IPHONE
    // A process here has a share of its own and is ended for exceeding it, so what the system has
    // free says nothing about what this process may still take: the kernel publishes the share's
    // remainder instead. 0 is "not published" (the simulator runs on the host's kernel and has no
    // share), and the host's own figure below is then the one that applies.
    if (const size_t left = os_proc_available_memory()) return (uint64_t) left;
#endif
    vm_statistics64_data_t vm;
    if (host_vm(&vm)) return host_available_pages(vm) * (uint64_t) vm_page();
#endif
    // Linux/Android: MemAvailable is the kernel's own estimate of what can be allocated without
    // swapping (it accounts for reclaimable page cache), which is exactly the sizing signal we want,
    // capped by what this process's control group still lets it take: in a container the machine's
    // free memory is not this process's to use (a rented box reported 103 GB available inside an
    // 85 GiB group).
    if (FILE * f = std::fopen("/proc/meminfo", "re")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                std::fclose(f);
                const uint64_t available = (uint64_t) kb * 1024ull;
                const uint64_t group = cgroup_memory_left();
                return group ? std::min(available, group) : available;
            }
        }
        std::fclose(f);
    }
    // Fallback where /proc is absent (the BSDs): free physical pages. An underestimate — it omits
    // reclaimable cache — but non-zero and safe to size a cache against.
#if defined(_SC_AVPHYS_PAGES)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long ps = sysconf(_SC_PAGESIZE);
    if (pages > 0 && ps > 0) return (uint64_t) pages * (uint64_t) ps;
#endif
    return 0;
}

uint64_t major_faults() {
    // ru_majflt counts faults that required a backing-store read (the ones that stall on flash).
    // RUSAGE_SELF aggregates every thread of the process, matching the multi-threaded decode.
    struct rusage ru;
    return getrusage(RUSAGE_SELF, &ru) == 0 ? (uint64_t) ru.ru_majflt : 0;
}
bool major_faults_counted() {
    return true;
}

double process_cpu_seconds() {
    // Total CPU consumed across all threads. Divided by wall×threads downstream, this is the
    // occupancy signal that tells a frequency cap / preemption apart from genuine heavy compute.
#if defined(CLOCK_PROCESS_CPUTIME_ID)
    struct timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == 0) return (double) ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
    return 0.0;
}

size_t fault_bytes() {
    // One major fault brings back one page. The kernel can fault a cluster in around a single miss,
    // so this is the floor of what a fault moved, not an upper bound — it under-reports rather than
    // inventing, which is the right direction for a number a reader will compare against real reads.
    return vm_page();
}

// Scan a "Key: <n> kB" file for the keys we want in one pass, rather than one open per field.
// Unused on Apple, which asks the kernel for the same figures (task_info, sysctl).
namespace {
[[maybe_unused]] bool scan_kb_file(const char * path, const char * const * keys, uint64_t * out, int n) {
    FILE * f = std::fopen(path, "re");
    if (!f) return false;
    char line[256];
    int found = 0;
    while (found < n && std::fgets(line, sizeof(line), f)) {
        for (int i = 0; i < n; ++i) {
            if (out[i]) continue; // already have it
            const size_t klen = std::strlen(keys[i]);
            if (std::strncmp(line, keys[i], klen) != 0 || line[klen] != ':') continue;
            unsigned long long kb = 0;
            if (std::sscanf(line + klen + 1, " %llu kB", &kb) == 1) {
                out[i] = (uint64_t) kb * 1024ull;
                ++found;
            }
            break;
        }
    }
    std::fclose(f);
    return found > 0;
}
} // namespace

namespace {
// One number from a cgroup file: a decimal, or "max" (no limit, reported as 0). False if unreadable.
bool read_cgroup_number(const std::string & path, uint64_t * out) {
    FILE * f = std::fopen(path.c_str(), "re");
    if (!f) return false;
    char buf[64] = {0};
    const bool got = std::fgets(buf, sizeof(buf), f) != nullptr;
    std::fclose(f);
    if (!got) return false;
    if (std::strncmp(buf, "max", 3) == 0) {
        *out = 0;
        return true;
    }
    char * end = nullptr;
    const unsigned long long v = std::strtoull(buf, &end, 10);
    if (end == buf) return false;
    *out = (uint64_t) v;
    return true;
}

// A key's value in a cgroup memory.stat ("inactive_file 12345"), 0 when absent.
uint64_t read_cgroup_stat(const std::string & path, const char * key) {
    FILE * f = std::fopen(path.c_str(), "re");
    if (!f) return 0;
    char line[256];
    const size_t klen = std::strlen(key);
    uint64_t v = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, key, klen) == 0 && line[klen] == ' ') {
            v = (uint64_t) std::strtoull(line + klen + 1, nullptr, 10);
            break;
        }
    }
    std::fclose(f);
    return v;
}

// A v1 limit at or above this is the kernel's "unlimited" (PAGE_COUNTER_MAX rounded to a page),
// not a cap anyone set.
constexpr uint64_t k_cgroup_v1_unlimited = 1ull << 62;

// Where this process sits: "0::<path>" for the unified hierarchy, "<n>:<controllers>:<path>" for a
// v1 hierarchy, of which the one naming `controller` is kept with its mount's directory name.
struct CgroupPlace {
    bool v2 = false, v1 = false;
    std::string v2_path, v1_path, v1_dir;
};
CgroupPlace cgroup_place(const char * self, const char * controller) {
    CgroupPlace c;
    FILE * f = std::fopen(self, "re");
    if (!f) return c;
    char line[1024];
    while (std::fgets(line, sizeof(line), f)) {
        std::string l(line);
        while (!l.empty() && (l.back() == '\n' || l.back() == '\r'))
            l.pop_back();
        if (l.rfind("0::", 0) == 0) {
            c.v2 = true;
            c.v2_path = l.substr(3);
        } else {
            const size_t a = l.find(':'), b = a == std::string::npos ? a : l.find(':', a + 1);
            if (b == std::string::npos) continue;
            const std::string controllers = l.substr(a + 1, b - a - 1);
            if (("," + controllers + ",").find("," + std::string(controller) + ",") != std::string::npos) {
                c.v1 = true;
                c.v1_path = l.substr(b + 1);
                c.v1_dir = controllers; // mounted as "cpu,cpuacct" with "cpu" a link to it
            }
        }
    }
    std::fclose(f);
    return c;
}

// Each directory from the group up to the root of what this namespace shows: a limit anywhere
// above applies too.
template <typename F> void cgroup_walk(const std::string & base, std::string path, F && visit) {
    for (;;) {
        visit(base + (path == "/" ? "" : path));
        if (path.empty() || path == "/") break;
        const size_t slash = path.find_last_of('/');
        path = slash == 0 || slash == std::string::npos ? "/" : path.substr(0, slash);
    }
}
} // namespace

uint64_t cgroup_memory_left(const char * fs_root, const char * self) {
    const CgroupPlace place = cgroup_place(self, "memory");
    const std::string root = fs_root;
    uint64_t best = 0; // the tightest finite remainder found
    auto consider = [&](uint64_t limit, uint64_t used, uint64_t reclaimable) {
        if (limit == 0) return;
        const uint64_t held = used > reclaimable ? used - reclaimable : 0;
        const uint64_t left = limit > held ? limit - held : 1; // over the cap: nothing left, but known
        if (best == 0 || left < best) best = left;
    };
    // Both file LRU lists are credited, as MemAvailable credits them: a model file read twice by the
    // last load sits on the active list, clean, and the kernel reclaims it before it would end
    // anything. Crediting the inactive list alone read a rented box's 30 GiB group as 10.
    auto walk = [&](const std::string & base, const std::string & path, const char * max_f, const char * cur_f,
                    const char * inactive_key, const char * active_key, bool v1_units) {
        cgroup_walk(base, path, [&](const std::string & dir) {
            uint64_t limit = 0, used = 0;
            if (read_cgroup_number(dir + "/" + max_f, &limit) && read_cgroup_number(dir + "/" + cur_f, &used)) {
                if (v1_units && limit >= k_cgroup_v1_unlimited) limit = 0;
                const std::string stat = dir + "/memory.stat";
                consider(limit, used, read_cgroup_stat(stat, inactive_key) + read_cgroup_stat(stat, active_key));
            }
        });
    };
    if (place.v2) walk(root, place.v2_path, "memory.max", "memory.current", "inactive_file", "active_file", false);
    if (place.v1)
        walk(root + "/memory", place.v1_path, "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file",
             "total_active_file", true);
    return best;
}

uint32_t cgroup_cpu_limit(const char * fs_root, const char * self) {
    const CgroupPlace place = cgroup_place(self, "cpu");
    const std::string root = fs_root;
    double best = 0.0; // the tightest quota found, in CPUs
    auto consider = [&](long long quota, long long period) {
        if (quota <= 0 || period <= 0) return; // "max" or v1's -1: no quota
        const double cpus = (double) quota / (double) period;
        if (best == 0.0 || cpus < best) best = cpus;
    };
    if (place.v2)
        cgroup_walk(root, place.v2_path, [&](const std::string & dir) {
            // "<quota> <period>", the quota "max" where none is set.
            FILE * f = std::fopen((dir + "/cpu.max").c_str(), "re");
            if (!f) return;
            char q[32] = {0};
            long long period = 0;
            if (std::fscanf(f, "%31s %lld", q, &period) == 2 && std::strcmp(q, "max") != 0)
                consider(std::strtoll(q, nullptr, 10), period);
            std::fclose(f);
        });
    if (place.v1)
        cgroup_walk(root + "/" + place.v1_dir, place.v1_path, [&](const std::string & dir) {
            long long quota = 0, period = 0;
            FILE * fq = std::fopen((dir + "/cpu.cfs_quota_us").c_str(), "re");
            FILE * fp = std::fopen((dir + "/cpu.cfs_period_us").c_str(), "re");
            if (fq && fp && std::fscanf(fq, "%lld", &quota) == 1 && std::fscanf(fp, "%lld", &period) == 1)
                consider(quota, period);
            if (fq) std::fclose(fq);
            if (fp) std::fclose(fp);
        });
    // A fraction of a CPU is still a thread that runs: 1.5 CPUs serve two, throttled.
    return best > 0.0 ? (uint32_t) std::ceil(best - 1e-9) : 0;
}

#if defined(__APPLE__)
// The task's own ledger carries the same split /proc/self/status does: internal pages are the
// anonymous ones, external the file-backed ones, and what the compressor holds is the anonymous
// memory already taken back — the role zram plays in VmSwap.
bool process_memory(ProcessMemory * out) {
    task_vm_info_data_t ti{};
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &ti, &n) != KERN_SUCCESS || n < TASK_VM_INFO_REV0_COUNT)
        return false;
    ProcessMemory sample;
    sample.rss_bytes = ti.resident_size;
    sample.rss_anon_bytes = ti.internal;
    sample.rss_file_bytes = ti.external;
    sample.swap_bytes = ti.compressed;
    // Older kernels can successfully return an earlier revision of this variable-length reply.
    if (n >= TASK_VM_INFO_REV1_COUNT) {
        sample.process_footprint_known = true;
        sample.process_footprint_bytes = ti.phys_footprint;
    }
#if TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR && !TARGET_OS_MACCATALYST
    // This build runs in an app on the device. Zero is a real exhausted allowance there; the
    // simulator has no app limit and must stay unknown. The API also returns zero for non-app
    // processes, which are not a supported execution context for the device engine.
    sample.process_headroom_known = true;
    sample.process_headroom_bytes = (uint64_t) os_proc_available_memory();
#endif
    *out = sample;
    return true;
}

bool device_memory(DeviceMemory * out) {
    vm_statistics64_data_t vm;
    if (!host_vm(&vm)) return false;
    out->available_bytes = host_available_pages(vm) * (uint64_t) vm_page();
    out->free_bytes = host_free_pages(vm) * (uint64_t) vm_page();
    // Swap files are created on demand, so "free" is what is left of the ones that exist now.
    struct xsw_usage sw;
    size_t len = sizeof(sw);
    out->swap_free_bytes = sysctlbyname("vm.swapusage", &sw, &len, nullptr, 0) == 0 ? sw.xsu_avail : 0;
    const uint64_t inactive = vm.inactive_count, file_backed = vm.external_page_count;
    out->idle_anon_bytes = (inactive > file_backed ? inactive - file_backed : 0) * (uint64_t) vm_page();
    out->compressed_stored_bytes = vm.total_uncompressed_pages_in_compressor * (uint64_t) vm_page();
    out->compressed_held_bytes = (uint64_t) vm.compressor_page_count * (uint64_t) vm_page();
    return true;
}

bool reclaim_facts(ReclaimFacts * out) {
#if TARGET_OS_IPHONE
    // The share's remainder is published only where there is a share: 0 on the simulator, which
    // runs on the host's kernel and is answered by the host's facts below.
    out->own_share_left = (uint64_t) os_proc_available_memory();
#endif
    vm_statistics64_data_t vm;
    if (!host_vm(&vm)) return out->own_share_left > 0;
    // Compressions since boot: a kernel that compresses has done so within seconds of starting.
    out->compresses = vm.compressions > 0;
    // Swap files are created on demand, so an empty total says nothing; the switch does.
    int enabled = 0;
    size_t len = sizeof(enabled);
    struct xsw_usage sw;
    size_t swlen = sizeof(sw);
    out->swaps = (sysctlbyname("vm.swap_enabled", &enabled, &len, nullptr, 0) == 0 && enabled != 0) ||
                 (sysctlbyname("vm.swapusage", &sw, &swlen, nullptr, 0) == 0 && sw.xsu_total > 0);
    return true;
}
#else
bool process_memory(ProcessMemory * out) {
    static const char * const keys[] = {"VmRSS", "RssAnon", "RssFile", "VmSwap"};
    uint64_t v[4] = {0, 0, 0, 0};
    if (!scan_kb_file("/proc/self/status", keys, v, 4)) return false;
    ProcessMemory sample;
    sample.rss_bytes = v[0];
    sample.rss_anon_bytes = v[1];
    sample.rss_file_bytes = v[2];
    sample.swap_bytes = v[3];
    // A control group's limit is the bytes left before the kernel ends this process, the same
    // fact an app's own share is where the platform keeps one.
    if (const uint64_t left = cgroup_memory_left()) {
        sample.process_headroom_known = true;
        sample.process_headroom_bytes = left;
    }
    *out = sample;
    return true;
}

bool device_memory(DeviceMemory * out) {
    static const char * const keys[] = {"MemAvailable", "MemFree", "SwapFree", "Inactive(anon)"};
    uint64_t v[4] = {0, 0, 0, 0};
    if (!scan_kb_file("/proc/meminfo", keys, v, 4)) return false;
    out->available_bytes = v[0];
    out->free_bytes = v[1];
    out->swap_free_bytes = v[2];
    out->idle_anon_bytes = v[3];
    // The compressed block device's own counters: bytes stored, then bytes they occupy. Unreadable
    // to an unprivileged process on some systems, which leaves both at 0.
    if (FILE * f = std::fopen("/sys/block/zram0/mm_stat", "re")) {
        unsigned long long stored = 0, held = 0;
        if (std::fscanf(f, "%llu %llu", &stored, &held) == 2) {
            out->compressed_stored_bytes = stored;
            out->compressed_held_bytes = held;
        }
        std::fclose(f);
    }
    return true;
}

// What the machine reclaims INTO. /proc/swaps would say directly, but an unprivileged process on
// Android may not read it (nor a zram device's own files). Two things it CAN see suffice: whether a
// zram block device EXISTS (the directory entry is visible even where its files are not) and
// whether /proc/meminfo reports any swap at all. A zram device that is swapped to means anonymous
// memory is compressed in RAM; swap without zram means it is written out.
//
// Any zram device, not the first: devices are numbered as they are added, and a machine that
// removed zram0 or configures several swaps on zram1.. compresses just the same. Where the listing
// is refused, the device the kernel creates first is still asked for by name.
bool reclaim_facts(ReclaimFacts * out) {
    static const char * const keys[] = {"SwapTotal"};
    uint64_t swap_total = 0;
    if (!scan_kb_file("/proc/meminfo", keys, &swap_total, 1)) return false;
    bool has_zram = false;
    if (DIR * d = ::opendir("/sys/block")) {
        while (const dirent * e = ::readdir(d)) {
            if (std::strncmp(e->d_name, "zram", 4) == 0) {
                has_zram = true;
                break;
            }
        }
        ::closedir(d);
    }
    if (!has_zram) {
        struct stat st {};
        has_zram = ::stat("/sys/block/zram0", &st) == 0;
    }
    out->swaps = swap_total > 0;
    out->compresses = has_zram && out->swaps;
    // A control group with a finite limit ends the process past it, whatever the machine could
    // still compress or swap: the group's remainder is this process's share.
    out->own_share_left = cgroup_memory_left();
    return true;
}
#endif

#if defined(__APPLE__) && TARGET_OS_IPHONE
// The address-space walk below is not offered to an app here, so the question goes unanswered and
// the caller reports the mapping's residency as unmeasured.
bool file_mapped_regions(const char *, std::vector<MappedRegion> &) {
    return false;
}
#elif defined(__APPLE__)
// Darwin has no /proc. The same question — which of this process's regions map this file, and from
// which file offset — is answered by walking its own address space: PROC_PIDREGIONPATHINFO returns
// the region containing or following an address together with the vnode path behind it, and asking
// about oneself needs no entitlement.
bool file_mapped_regions(const char * basename, std::vector<MappedRegion> & out) {
    const size_t blen = std::strlen(basename);
    const pid_t pid = getpid();
    bool any = false;
    uint64_t addr = 0;
    for (;;) {
        struct proc_regionwithpathinfo info;
        if (proc_pidinfo(pid, PROC_PIDREGIONPATHINFO, addr, &info, sizeof(info)) != (int) sizeof(info)) break;
        const uint64_t start = info.prp_prinfo.pri_address;
        const uint64_t size = info.prp_prinfo.pri_size;
        if (size == 0 || start + size <= addr) break; // no forward progress: the walk is over
        addr = start + size;
        const char * path = info.prp_vip.vip_path;
        const size_t plen = strnlen(path, sizeof(info.prp_vip.vip_path));
        if (plen < blen || std::strncmp(path + plen - blen, basename, blen) != 0) continue;
        out.push_back({(uintptr_t) start, (uintptr_t) (start + size), info.prp_prinfo.pri_offset});
        any = true;
    }
    return any;
}
#else
bool file_mapped_regions(const char * basename, std::vector<MappedRegion> & out) {
    FILE * f = std::fopen("/proc/self/maps", "re");
    if (!f) return false;
    const size_t blen = std::strlen(basename);
    char line[512];
    bool any = false;
    // Each line: "start-end perms offset dev inode   pathname". We want the VMAs whose pathname ends
    // with the model's file name — an mmap of a large file appears as one or more such VMAs.
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long long start = 0, end = 0, off = 0;
        // The pathname is the last field; sscanf %n gives us where the fixed part ended so we can
        // scan the remainder for it without copying.
        int consumed = 0;
        if (std::sscanf(line, "%llx-%llx %*s %llx %*s %*s %n", &start, &end, &off, &consumed) < 3) continue;
        const char * path = line + consumed;
        // Trim the trailing newline and any leading spaces %n may have left.
        while (*path == ' ')
            ++path;
        size_t plen = std::strlen(path);
        while (plen && (path[plen - 1] == '\n' || path[plen - 1] == ' '))
            --plen;
        if (plen < blen) continue;
        if (std::strncmp(path + plen - blen, basename, blen) != 0) continue;
        out.push_back({(uintptr_t) start, (uintptr_t) end, (uint64_t) off});
        any = true;
    }
    std::fclose(f);
    return any;
}
#endif

#endif

// ── A second process, for work that can end one ─────────────────────────────────────────
#if defined(_WIN32)

namespace {
// Between UTF-8, which every path here is written in, and what the system's own calls take.
std::wstring widen(const std::string & s, UINT code_page) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(code_page, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) (n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(code_page, 0, s.data(), (int) s.size(), w.data(), n);
    return w;
}
std::string narrow_utf8(const wchar_t * w, size_t len) {
    if (len == 0) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, (int) len, nullptr, 0, nullptr, nullptr);
    std::string s((size_t) (n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w, (int) len, s.data(), n, nullptr, nullptr);
    return s;
}
} // namespace

// Asked of the system in its own wide characters and returned as UTF-8. Asked in the narrow form, a
// folder named in the system's code page came back in that code page's bytes, and the first reader
// that took them for UTF-8 (the listing of the backends beside the executable) threw: a release
// unpacked under such a folder ended at start (#677).
std::string self_path() {
    wchar_t buf[4096];
    const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) (sizeof(buf) / sizeof(buf[0])));
    if (n == 0 || n >= sizeof(buf) / sizeof(buf[0])) return "";
    return without_links(narrow_utf8(buf, (size_t) n));
}

bool can_run_self() {
    return !self_path().empty();
}

namespace {
// One argument, quoted the way the C runtime of the child reads it back: a quote is escaped, and
// backslashes are literal except the run that precedes a quote or the end, which is doubled.
void append_quoted(std::string & cmd, const std::string & a) {
    cmd += ' ';
    cmd += '"';
    size_t bs = 0;
    for (char c : a) {
        if (c == '\\') {
            ++bs;
            continue;
        }
        if (c == '"') {
            cmd.append(bs * 2 + 1, '\\');
        } else {
            cmd.append(bs, '\\');
        }
        bs = 0;
        cmd += c;
    }
    cmd.append(bs * 2, '\\');
    cmd += '"';
}

// Whatever the pipe holds right now, appended to the tail kept for the caller.
bool drain_pipe(HANDLE rd, std::string * out, size_t keep) {
    for (;;) {
        DWORD avail = 0;
        if (!PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr)) return false; // the child closed its end
        if (avail == 0) return true;
        char buf[8192];
        DWORD want = avail < (DWORD) sizeof(buf) ? avail : (DWORD) sizeof(buf);
        DWORD n = 0;
        if (!ReadFile(rd, buf, want, &n, nullptr) || n == 0) return false;
        if (out) {
            out->append(buf, (size_t) n);
            if (out->size() > 2 * keep) out->erase(0, out->size() - keep);
        }
    }
}
} // namespace

ChildResult run_self(const std::vector<std::string> & args, double timeout_seconds, std::string * out) {
    ChildResult r;
    const std::string path = self_path();
    if (path.empty()) return r;

    // One pipe carries the child's stdout and stderr; only the parent's end stays uninherited.
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return r;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    std::string cmd;
    append_quoted(cmd, path);
    for (const std::string & a : args)
        append_quoted(cmd, a);
    // The arguments are in the encoding this process was given its own in (its code page, UTF-8
    // where the executable's manifest says so); the executable's path is UTF-8 whatever that is.
    std::wstring cmdline = widen(cmd.substr(1), CP_ACP); // without the leading space
    const std::wstring wpath = widen(path, CP_UTF8);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    const BOOL started = CreateProcessW(wpath.c_str(), cmdline.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                                        nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (!started) {
        CloseHandle(rd);
        return r;
    }

    // Read until the child closes its end or its time is up. The pipe is polled rather than read
    // blocking, because a blocking read has no deadline and a child that never answers is the
    // case this exists for.
    const size_t keep = 1u << 20;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);
    bool timed_out = false;
    for (;;) {
        if (!drain_pipe(rd, out, keep)) break;
        const double left = std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0.0) {
            timed_out = true;
            break;
        }
        const DWORD slice = left * 1000.0 < 100.0 ? (DWORD) (left * 1000.0) + 1 : 100;
        if (WaitForSingleObject(pi.hProcess, slice) == WAIT_OBJECT_0) {
            drain_pipe(rd, out, keep); // what it wrote last
            break;
        }
    }
    CloseHandle(rd);
    if (timed_out) TerminateProcess(pi.hProcess, 1);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (timed_out) {
        r.outcome = ChildOutcome::TimedOut;
    } else if (code >= 0xC0000000u) {
        // An NTSTATUS: the child died of an exception (an access violation is 0xC0000005), which is
        // this platform's way of saying what a signal says elsewhere.
        r.outcome = ChildOutcome::Signalled;
        r.code = (int) code;
    } else {
        r.outcome = ChildOutcome::Exited;
        r.code = (int) code;
    }
    return r;
}

#else

} // namespace bmoe::pio
extern char ** environ;
namespace bmoe::pio {

std::string self_path() {
#if defined(__APPLE__)
    char buf[4096];
    uint32_t n = sizeof(buf);
    if (_NSGetExecutablePath(buf, &n) != 0) return "";
    return without_links(buf);
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    buf[n] = 0;
    return without_links(buf);
#endif
}

bool can_run_self() {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    // An app may not start a process, its own executable included: the spawn is refused, and a
    // refusal would read as a device whose probe could not be run. Say so before anyone asks.
    return false;
#elif defined(__ANDROID__)
    // LLM Hub: inside an Android app the executable is the system's app_process, not a bmoe
    // binary, so there is no self to run (and posix_spawn needs API 28; the app supports 27).
    return false;
#else
    return !self_path().empty();
#endif
}

ChildResult run_self(const std::vector<std::string> & args, double timeout_seconds, std::string * out) {
    ChildResult r;
#if defined(__ANDROID__)
    // LLM Hub: see can_run_self(); a caller does the work in its own process instead.
    (void) args;
    (void) timeout_seconds;
    (void) out;
    return r;
#else
    const std::string path = self_path();
    int fds[2];
    if (path.empty() || pipe(fds) != 0) return r;

    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(path.c_str()));
    for (const std::string & a : args)
        argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 2);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addclose(&fa, fds[1]);
    pid_t pid = 0;
    const int rc = posix_spawn(&pid, path.c_str(), &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (rc != 0) {
        close(fds[0]);
        return r;
    }

    // Read until the child closes its end or its time is up. Only the tail is kept: the line a
    // caller wants is the last thing a well-behaved child prints, and a backend can be chatty.
    const size_t keep = 1u << 20;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);
    bool timed_out = false;
    for (;;) {
        const double left = std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0.0) {
            timed_out = true;
            break;
        }
        struct pollfd pf {
            fds[0], POLLIN, 0
        };
        const int pr = poll(&pf, 1, (int) std::min(left * 1000.0, 1000.0));
        if (pr < 0) break;
        if (pr == 0) continue;
        char buf[8192];
        const ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n <= 0) break; // the child closed its end
        if (out) {
            out->append(buf, (size_t) n);
            if (out->size() > 2 * keep) out->erase(0, out->size() - keep);
        }
    }
    close(fds[0]);
    if (timed_out) kill(pid, SIGKILL);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (timed_out) {
        r.outcome = ChildOutcome::TimedOut;
    } else if (WIFSIGNALED(status)) {
        r.outcome = ChildOutcome::Signalled;
        r.code = WTERMSIG(status);
    } else {
        r.outcome = ChildOutcome::Exited;
        r.code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    return r;
#endif
}

#endif

// ── Cache-bypass read semantics ─────────────────────────────────────────────────────────
bool direct_needs_alignment() {
#if defined(__APPLE__)
    return false; // F_NOCACHE is a caching hint, not an I/O mode: plain preads stay fully general
#else
    return true; // O_DIRECT / FILE_FLAG_NO_BUFFERING reject unaligned windows
#endif
}

namespace {

// The smallest granule a direct read is cut to. No storage has a sector below 512 bytes, and an
// aligned allocation smaller than a pointer is not one posix_memalign accepts.
constexpr uint64_t direct_align_min = 512;

// A stated alignment as a granule the readers can mask with: a power of two, raised to the floor;
// 0 when the value is not one.
[[maybe_unused]] size_t as_granule(uint64_t a) {
    if (a == 0 || (a & (a - 1)) != 0) return 0;
    return (size_t) (a < direct_align_min ? direct_align_min : a);
}

} // namespace

#if defined(_WIN32)
size_t direct_alignment(fd_t fd) {
#if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0602 // FileStorageInfo: Windows 8 and later
    FILE_STORAGE_INFO si;
    memset(&si, 0, sizeof(si));
    if (!fd_ok(fd) || !GetFileInformationByHandleEx((HANDLE) fd, FileStorageInfo, &si, sizeof(si))) return 0;
    // NO_BUFFERING wants offsets and lengths in whole logical sectors and buffers on the physical
    // sector the filesystem works in; one granule must satisfy both, so it is the larger.
    uint64_t a = (uint64_t) si.LogicalBytesPerSector;
    if ((uint64_t) si.FileSystemEffectivePhysicalBytesPerSectorForAtomicity > a)
        a = (uint64_t) si.FileSystemEffectivePhysicalBytesPerSectorForAtomicity;
    return as_granule(a);
#else
    (void) fd;
    return 0;
#endif
}
#else
namespace {

// What the kernel states as this file's direct-I/O alignment, or 0 where it cannot be asked. Issued
// by syscall number: bionic declares the wrapper only from API 30 and glibc only from 2.28, and
// the number is what both of them call.
size_t statx_dio_align(int fd) {
#if defined(__linux__) && defined(__NR_statx) && defined(STATX_DIOALIGN) && defined(AT_EMPTY_PATH)
#if defined(__ANDROID__)
    // An app's sandbox admits the syscalls bionic itself issues, and bionic issues statx from API
    // 30. Below that the call is not refused but trapped, and the trap ends the process.
    if (android_get_device_api_level() < 30) return 0;
#endif
    struct statx stx;
    memset(&stx, 0, sizeof(stx));
    if (syscall(__NR_statx, fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &stx) != 0) return 0;
    // A kernel older than the field leaves the mask bit clear; an offset alignment of 0 is the
    // kernel saying this file takes no direct I/O at all. Either way, it did not answer.
    if (!(stx.stx_mask & STATX_DIOALIGN) || stx.stx_dio_offset_align == 0) return 0;
    uint64_t a = (uint64_t) stx.stx_dio_offset_align;
    if ((uint64_t) stx.stx_dio_mem_align > a) a = (uint64_t) stx.stx_dio_mem_align;
    return as_granule(a);
#else
    (void) fd;
    return 0;
#endif
}

// The largest block size a local filesystem formats with (ext4, XFS and btrfs stop at 64 KiB; f2fs
// uses 4 or 16 KiB). A larger "block size" is a transfer-size hint - network and FUSE filesystems report
// their read size in that field - and taking it as a granule would only inflate every window.
constexpr uint64_t direct_align_block_max = 64 * 1024;

// A filesystem's block size taken as a direct-read granule, or 0 when the field is not one.
size_t block_granule(uint64_t bsize) {
    return bsize > direct_align_block_max ? 0 : as_granule(bsize);
}

} // namespace

size_t direct_alignment(fd_t fd) {
    if (!fd_ok(fd)) return 0;
    // The kernel's answer is the legality floor: a window aligned to a multiple of it is legal too.
    // Reading at exactly 512 bytes where it allows that (ext4 on a 512-byte-sector NVMe) is a
    // throughput change on the very SSDs this project has measured punishing small reads, and it
    // has not been measured, so the granule stays at least the value every direct read used before
    // the alignment was asked for. A measured win is what moves this line, not the probe.
    if (const size_t a = statx_dio_align(fd)) return a < direct_align_fallback ? direct_align_fallback : a;
    // The filesystem's fundamental block (f_frsize; f_bsize is the preferred transfer size on some
    // systems, Apple's among them). Not what O_DIRECT is specified against - that is the device's
    // logical block, or the filesystem's own where it is stricter (f2fs) - but a filesystem block
    // is a whole number of device blocks, so it is the stricter of the two.
    struct statvfs sv;
    memset(&sv, 0, sizeof(sv));
    if (fstatvfs(fd, &sv) != 0) return 0;
    const size_t b = block_granule((uint64_t) (sv.f_frsize ? sv.f_frsize : sv.f_bsize));
    if (b == 0) return 0;
#if defined(__APPLE__)
    return b; // no alignment contract to fall short of: the block is reported as it is
#else
    // Never under the fallback on a block size alone: a granule too large costs a little over-read,
    // one too small fails every read, and the block size is weaker evidence than the 4096 every
    // direct read has used so far. Only the kernel's own answer above may go under it.
    return b < direct_align_fallback ? direct_align_fallback : b;
#endif
}
#endif

// ── Reclaim-exempt allocation ────────────────────────────────────────────────────────────
// One section for every platform: Android and Darwin have such a store, everywhere else this
// reports "unsupported" and callers fall back to an ordinary allocation. Declared in the header with the measured
// properties and the reason the ceiling is a lock boundary rather than an allocation one.
LockLimit intersect_lock_limits(LockLimit a, LockLimit b) {
    if ((a.known && a.bytes == 0) || (b.known && b.bytes == 0)) return {0, true};
    return {std::min<uint64_t>(a.bytes, b.bytes), a.known && b.known};
}

LockLimit remaining_lock_limit(LockLimit limit, uint64_t held, bool held_known) {
    if (limit.known && limit.bytes == 0) return limit;
    if (limit.bytes == lock_unbounded) return limit;
    if (!held_known) return {limit.bytes, false};
    return {limit.bytes > held ? limit.bytes - held : 0, limit.known};
}

uint64_t lockable_bytes() {
    return lockable_limit().bytes;
}

uint64_t lock_in_place_bytes() {
    return lock_in_place_limit().bytes;
}

#if !defined(_WIN32)
namespace {
// What this process may lock in place, as the kernel's own per-process limit states it.
LockLimit memlock_limit() {
    struct rlimit r {};
    if (getrlimit(RLIMIT_MEMLOCK, &r) != 0) return {};
    return {r.rlim_cur == RLIM_INFINITY ? lock_unbounded : (uint64_t) r.rlim_cur, true};
}
} // namespace
#endif

#if defined(__ANDROID__)

namespace {
// AHardwareBuffer is API 26. A build whose minimum is older (Termux's toolchain targets 24) binds it
// weakly (core/CMakeLists.txt), and the attribute makes a use of these two outside the check in
// ahwb() a build error.
__attribute__((availability(android, introduced = 26))) bool ahwb_alloc(size_t sz, PinnedAlloc * out) {
    AHardwareBuffer_Desc d{};
    d.width = (uint32_t) sz; // BLOB: width IS the byte count, and height must be 1
    d.height = 1;
    d.layers = 1;
    d.format = AHARDWAREBUFFER_FORMAT_BLOB;
    // CPU_READ_OFTEN asks gralloc for a cacheable mapping. Measured: the hint makes no difference
    // here (CPU_READ_RARELY reads identically), but asking for what we actually do is still right —
    // another gralloc may honour it.
    d.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;

    AHardwareBuffer * buf = nullptr;
    if (AHardwareBuffer_allocate(&d, &buf) != 0 || !buf) return false;
    void * p = nullptr;
    if (AHardwareBuffer_lock(buf, d.usage, -1, nullptr, &p) != 0 || !p) {
        AHardwareBuffer_release(buf);
        return false;
    }
    out->base = p;
    out->handle = buf;
    out->size = sz;
    return true;
}

__attribute__((availability(android, introduced = 26))) void ahwb_release(PinnedAlloc * a) {
    AHardwareBuffer * buf = (AHardwareBuffer *) a->handle;
    AHardwareBuffer_unlock(buf, nullptr);
    AHardwareBuffer_release(buf);
}

struct Ahwb {
    bool (*alloc)(size_t, PinnedAlloc *);
    void (*release)(PinnedAlloc *);
};

// The one place that asks the device. Below API 26 there is no such store: null, and every function
// below answers as the last branch does, the same "absent" the plan reads on a platform without
// one. Where the minimum is 26 or more, the check folds to true.
const Ahwb * ahwb() {
    if (__builtin_available(android 26, *)) {
        static constexpr Ahwb calls{ahwb_alloc, ahwb_release};
        return &calls;
    }
    return nullptr;
}
} // namespace

// A dma-buf is charged to no limit this process can read: gralloc publishes no total, so the
// answer is "unbounded here" and the caller bounds it by what the process may hold. Locking in
// place is a different store with a different answer - the vendor caps it at a few pages.
LockLimit lockable_limit() {
    if (!ahwb()) return {0, true};
    return {lock_unbounded, true};
}
LockLimit lock_in_place_limit() {
    return memlock_limit();
}

size_t pinned_max_bytes() {
    // The lock path uses a signed 32-bit type: AHardwareBuffer_lock returns EINVAL at exactly 2^31
    // bytes while 2^31 - 1 succeeds, even though allocation reaches the 32-bit width cap of 4 GiB.
    // Probing this at runtime would cost a multi-GiB allocation at startup to learn a constant, so
    // it is stated here and verified by `bmoe-membench --probe-max` instead.
    if (!ahwb()) return 0;
    return 0x7FFFFFFFu;
}

bool pinned_alloc(size_t sz, PinnedAlloc * out) {
    const Ahwb * calls = ahwb();
    if (!calls || !out || sz == 0 || sz > pinned_max_bytes()) return false;
    return calls->alloc(sz, out);
}

void pinned_free(PinnedAlloc * a) {
    const Ahwb * calls = ahwb();
    if (!calls || !a || !a->handle) return; // a handle comes only from pinned_alloc
    calls->release(a);
    a->base = nullptr;
    a->handle = nullptr;
    a->size = 0;
}

#elif defined(__APPLE__)

namespace {
LockLimit wire_limit(const char * name) {
    uint64_t limit = 0;
    size_t len = sizeof(limit);
    if (sysctlbyname(name, &limit, &len, nullptr, 0) != 0) return {};
    return {limit, true};
}
} // namespace

// Darwin's reclaim-exempt store is wired memory: an anonymous mapping the process has locked, which
// the kernel neither swaps nor compresses. Any process may ask, up to a system-wide limit that is
// most of RAM, so the limit is read rather than assumed.
size_t pinned_max_bytes() {
    const LockLimit limit = wire_limit("vm.user_wire_limit");
    return limit.known ? (size_t) limit.bytes : 0;
}
// What is left of that limit, and the limit is GLOBAL: it bounds the wired memory of the whole
// system, the kernel's own included, not this process's share of it. Measured by locking anonymous
// memory in 32 MiB steps until the kernel refused: 10336 MiB granted where the limit less the
// system's wired count stood at 10368, on a machine whose limit is 12451. So a device that wires
// its buffers, or any other process that locks memory, shrinks this without this process having
// allocated anything - which is why it is read each time rather than derived from the limit.
LockLimit lockable_limit() {
    const LockLimit global = wire_limit("vm.global_user_wire_limit");
    vm_statistics64_data_t vm{};
    const bool usage_known = host_vm(&vm);
    const uint64_t wired = usage_known ? (uint64_t) vm.wire_count * (uint64_t) vm_page() : 0;
    const LockLimit left = remaining_lock_limit(global, wired, usage_known);
    // These queries can be denied independently. Keep every finite bound that did answer without
    // turning a denied sysctl into a zero grant; mlock remains the authority on an actual request.
    return intersect_lock_limits(left, intersect_lock_limits(wire_limit("vm.user_wire_limit"), memlock_limit()));
}
// One store: a wired anonymous mapping is the same thing whether it was locked when it was
// allocated or afterwards.
LockLimit lock_in_place_limit() {
    return lockable_limit();
}
bool pinned_alloc(size_t sz, PinnedAlloc * out) {
    if (!sz || !out) return false;
    void * p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return false;
    if (mlock(p, sz) != 0) { // over the limit: say so by failing, as a full device would
        munmap(p, sz);
        return false;
    }
    out->base = p;
    out->handle = nullptr;
    out->size = sz;
    return true;
}
void pinned_free(PinnedAlloc * a) {
    if (!a || !a->base) return;
    munmap(a->base, a->size);
    *a = PinnedAlloc{};
}

#else

size_t pinned_max_bytes() {
    return 0;
}
// No reclaim-exempt allocation here, so nothing to total. Locking in place is whatever the
// process limit says, and on Windows nothing: VirtualLock is bounded by a working-set quota that
// is not ours to raise, which is why vm_pin declines there.
LockLimit lockable_limit() {
    return {0, true};
}
LockLimit lock_in_place_limit() {
#if defined(_WIN32)
    return {0, true};
#else
    return memlock_limit();
#endif
}
bool pinned_alloc(size_t, PinnedAlloc *) {
    return false;
}
void pinned_free(PinnedAlloc *) {}

#endif

bool read_through_cache(const char * path,
                        const std::vector<std::pair<uint64_t, uint64_t>> & ranges,
                        const std::atomic<bool> * stop) {
    const fd_t fd = open_read(path, /*direct*/ false);
    if (!fd_ok(fd)) return false;
    std::vector<char> buf(read_through_piece);
    bool ok = true;
    for (const auto & r : ranges) {
        for (uint64_t at = r.first, end = r.first + r.second; ok && at < end;) {
            if (stop && stop->load(std::memory_order_relaxed)) {
                close_fd(fd);
                return true;
            }
            const size_t n = (size_t) (std::min)(end - at, (uint64_t) buf.size());
            const long long got = pread_at(fd, buf.data(), n, at);
            if (got <= 0) ok = false;
            at += got > 0 ? (uint64_t) got : 0;
        }
    }
    close_fd(fd);
    return ok;
}

} // namespace bmoe::pio
