// Cross-platform positioned I/O and reserved virtual memory.
//
// Two primitive families the expert streamer is built on:
//
//   * Positioned reads with optional cache bypass (O_DIRECT on POSIX,
//     FILE_FLAG_NO_BUFFERING on Windows). Reads are done on aligned windows into an
//     aligned bounce buffer; the caller memcpy's the valid interior out.
//
//   * Reserve/commit/evict/release of virtual address ranges. The LRU expert cache
//     reserves a full-size address range per (layer, projection) but commits physical
//     pages only for the expert slices actually cached, and hands them back on
//     eviction — so a huge reserved span costs RAM only for resident experts.
//
// POSIX commits lazily (commit is a no-op, MADV_DONTNEED reclaims); Windows commits and
// decommits explicitly. Both honour a page-granularity that the caller rounds to.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h> // _mm_pause, for cpu_relax
#endif

namespace bmoe::pio {

#if defined(_WIN32)
using fd_t = void *; // HANDLE
#else
using fd_t = int;
#endif

extern const fd_t fd_invalid;
bool fd_ok(fd_t fd);

// Open path for positioned reads. When direct is true, request cache-bypassing I/O — O_DIRECT on
// Linux/Android, FILE_FLAG_NO_BUFFERING on Windows, F_NOCACHE on Apple. The descriptor itself is
// valid either way; *effective_direct (when not null) reports whether the cache-bypassing mode is
// actually in effect for it, because every platform can decline the request without failing the
// open — a caller that cares must ask here, never reconstruct the answer from what it passed.
// The caller should be prepared to reopen with direct=false for a sub-alignment tail.
fd_t open_read(const char * path, bool direct, bool * effective_direct = nullptr);
void close_fd(fd_t fd);
// Positioned blocking read. Returns bytes read, 0 at EOF, -1 on error.
long long pread_at(fd_t fd, void * buf, size_t count, uint64_t off);
uint64_t file_size(fd_t fd);

// Whether the platform's cache-bypassing mode constrains the read mechanics: O_DIRECT and
// FILE_FLAG_NO_BUFFERING reject unaligned offsets, lengths and buffers, so reads must go through
// aligned windows into an aligned bounce. Apple's F_NOCACHE — the only uncached mode there, since
// the platform has no O_DIRECT — is a per-descriptor hint to turn data caching off: no alignment
// contract, no DMA promise. "Uncached" (a property of the descriptor, what open_read reports) and
// "alignment-constrained" (a property of the I/O mode, what this reports) are the same thing on
// every platform except Apple, where they come apart.
bool direct_needs_alignment();

// The granule a cache-bypassing read on `fd` must be aligned to - its file offset, its length and
// its buffer, taken as one value - as the platform states it for the filesystem this file is on.
// It is a fact about the file, not the platform: one phone formats its data partition with 4 KiB
// blocks and the next with 16 KiB ones, and O_DIRECT on the second rejects a 4 KiB window. Asked of
// the descriptor the reads will use, so it answers for the path they actually take:
//
//   * Linux/Android: statx(STATX_DIOALIGN), which is the kernel stating exactly this, where the
//     kernel, the headers and (on Android) the app sandbox all have it; otherwise the filesystem's
//     fundamental block size. Neither goes below `direct_align_fallback`: a smaller legal granule
//     is a read-size change that has not been measured, not a correctness gain.
//   * Apple: the filesystem's fundamental block size. F_NOCACHE imposes no alignment at all (see
//     direct_needs_alignment), so this is a safe granule to report, not a requirement.
//   * Windows: the volume's logical sector size and the filesystem's effective physical sector size,
//     the larger of the two - NO_BUFFERING requires the first and recommends the second for buffers.
//
// A power of two, or 0 where the platform does not say; the caller then uses `direct_align_fallback`,
// and must not report that substitute as a measurement.
size_t direct_alignment(fd_t fd);
// What an unprobed file is read at: the block size of every flash and disk this engine has met, and
// the value every direct read used before the alignment was asked for. A multiple of the 512-byte
// sector, so it stays correct on storage that wants less.
constexpr size_t direct_align_fallback = 4096;

// The span two independently written atomics must keep apart so they never share a cache line:
// 128 bytes on Apple's cores, 64 on x86 and the other arm64 cores this engine has run on. It
// sizes the padding of the hottest poll in the engine, one flag per expert and projection, so it
// follows the target rather than taking the widest value everywhere: doubling that array's
// footprint on a phone is a cost that was not measured. A compile-time value because it sizes
// `alignas`.
#if defined(__APPLE__) && defined(__aarch64__)
constexpr size_t false_sharing_span = 128;
#else
constexpr size_t false_sharing_span = 64;
#endif

// One idle beat of a spin-wait: keep the core out of its siblings' way without entering the
// scheduler. A pause hint where the ISA has one (single instructions, no syscall); yield elsewhere.
inline void cpu_relax() {
#if defined(__aarch64__)
    __asm__ __volatile__("isb" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
    _mm_pause();
#else
    __builtin_ia32_pause();
#endif
#else
    std::this_thread::yield();
#endif
}

// Aligned heap allocation for O_DIRECT bounce buffers and shared slots.
void * alloc_aligned(size_t align, size_t sz);
void aligned_free(void * p);

// A reclaim-exempt allocation: memory the kernel may not take back under pressure.
//
// On Android this is a dma-buf, obtained as an AHardwareBuffer of format BLOB and locked for CPU
// access. Its pages are pinned for the buffer's lifetime because a device may DMA from them at any
// moment, which makes it the ONLY allocation an unprivileged app can make that reclaim cannot touch:
// `mlock` is capped at 64 KiB by the vendor, the cgroup protections are v2-only, and MADV_COLD merely
// redirects reclaim. See docs/android-memory.md.
//
// Measured properties (docs/bench-data/2026-07-21-pinned-memory/): reads at the same speed as
// ordinary anonymous memory — the failure mode that would have made this useless, an uncached
// mapping, does not occur — and costs a few hundred ms per GiB to allocate. Do not assume the
// bandwidth result holds on every gralloc; it is one device's.
//
// On Darwin it is wired memory: an anonymous mapping the process has locked, granted to any process
// up to a system-wide total (see lockable_bytes).
//
// `pinned_max_bytes()` is the reported ceiling for one allocation: 0 where the platform has no such
// allocation (Windows and Linux builds) or cannot report its ceiling. The latter does not establish
// that vm_pin is unavailable. On Android it is the LOCK boundary rather than the
// allocation one — AHardwareBuffer_lock fails with EINVAL at 2^31 bytes even though allocation
// reaches the 4 GiB format cap, so a larger working set must be split across several buffers.
struct PinnedAlloc {
    void * base = nullptr;   // CPU-readable pointer, or null if the allocation failed
    void * handle = nullptr; // opaque platform handle, released by pinned_free
    size_t size = 0;
};
size_t pinned_max_bytes();
bool pinned_alloc(size_t sz, PinnedAlloc * out);
void pinned_free(PinnedAlloc * a);

// How much this process may still hold reclaim-exempt in total: `lockable_bytes` through
// pinned_alloc, `lock_in_place_bytes` through vm_pin. Both are what is LEFT, net of what is already
// held against the same limit, so they move while the process runs and are read when a plan is
// made. 0 means the platform grants none; `lock_unbounded` means there is no reported total of its
// own and the caller must bound it by what the process may hold at all. The query facts distinguish
// a reported unlimited value from a query the sandbox denied: neither is evidence of a finite grant.
constexpr uint64_t lock_unbounded = ~0ull;
struct LockLimit {
    uint64_t bytes = lock_unbounded;
    // False if an applicable query failed, unless another established zero makes the answer
    // certain. `bytes` still preserves every independently known upper bound; the caller must not
    // report an uncertain bound as a measured grant.
    bool known = false;
};
LockLimit intersect_lock_limits(LockLimit a, LockLimit b);
LockLimit remaining_lock_limit(LockLimit limit, uint64_t held, bool held_known);
LockLimit lockable_limit();
LockLimit lock_in_place_limit();
// Upper bounds only; use the facts above wherever an unknown must remain distinguishable.
uint64_t lockable_bytes();
uint64_t lock_in_place_bytes();

// Run this same executable again as a child, with `args`, and wait at most `timeout_seconds` for
// it. Its stdout and stderr come back together in `out` (the tail of them, if there is a great
// deal). This exists for one job: doing something that can end a process - asking a device driver
// to work - somewhere other than here. A child that outlives its time is killed, and how it ended
// is reported rather than interpreted.
//
// `can_run_self()` is false where this platform has no implementation, and `run_self` then
// reports CouldNotStart; a caller falls back to doing the work in its own process.
enum class ChildOutcome { CouldNotStart, Exited, Signalled, TimedOut };
struct ChildResult {
    ChildOutcome outcome = ChildOutcome::CouldNotStart;
    int code = 0; // the exit status, or the signal number
};
bool can_run_self();
ChildResult run_self(const std::vector<std::string> & args, double timeout_seconds, std::string * out);
// The file the kernel started this process from (not argv[0], which is whatever the caller
// typed), every link on the way resolved, or "" where the platform does not say. Its directory is
// where a release's backend modules are looked for, so a link to the executable from another
// folder (install.sh puts one on PATH) still finds them (#605).
std::string self_path();

// Reserved (address-only) region; physical pages appear on commit, vanish on evict.
size_t vm_page();
// Readable bytes every buffer we allocate to hold tensor data keeps past the tensor's last byte.
// ggml's CPU kernels may load a vector that reaches beyond the last block (one Q6_K path reads 32
// bytes from a 16-byte field at the block's end), which llama.cpp's own buffers absorb because
// something always follows the last tensor in them. A buffer sized to its tensor exactly can end
// on a page boundary with a guard page after it, and that load faults - seen on a phone, at random,
// as the next mapping varied. One page, the OS's own, covers any vector a CPU loads at once.
inline size_t tensor_tail_slack() {
    return vm_page();
}
void * vm_reserve(size_t sz);
bool vm_commit(void * p, size_t sz);
// Ask the kernel to keep a committed span in RAM: not swapped, not compressed. False where the
// platform refuses (a locked-memory limit, or no such call), and the span is then ordinary memory.
// vm_evict releases a pinned span like any other. Whether this is available is the whole
// difference between platforms: macOS grants it to any process up to most of RAM, Android caps it
// at 64 KiB, and where it is granted it is what stops a cache hit from paying a decompression.
bool vm_pin(void * p, size_t sz);
void vm_evict(void * p, size_t sz);
void vm_release(void * p, size_t sz);

// Discardable spans: memory the kernel may take back by DROPPING it, never by writing it anywhere.
// Anonymous memory that is merely unpinned is swapped or compressed under pressure, and swap is a
// write to the drive; memory that is pinned makes the kernel swap everything else instead. A cache
// whose contents can be read again from the model file needs neither: offered pages are the first
// the kernel reclaims, before it compresses or swaps anything, and a reclaimed page costs one read.
//
// Every platform has the mechanism; each spells it differently, which is all these three hide:
//  - Darwin: a purgeable VM object per span (VM_PURGABLE_VOLATILE / NONVOLATILE). A state change is
//    per object, not per page, and taking the span back reports whether the kernel emptied it.
//  - Linux and Android: MADV_FREE. Nothing reports a loss, but a dropped page reads back as zero and
//    a write cancels a pending drop, so vm_reclaim writes every page first and then checks it.
//  - Windows: OfferVirtualMemory / ReclaimVirtualMemory, which reports a loss itself.
//
// vm_commit_discardable commits a page-aligned span so that it can later be offered (on Darwin the
// span must be one object, so it replaces whatever was mapped there; elsewhere it is vm_commit).
// vm_offer hands a committed span over; false where the platform cannot, and the span stays
// ordinary memory. vm_reclaim takes an offered span back before anything reads it: true when every
// page is still what was offered, false when any was dropped, and then the WHOLE span is ordinary,
// committed, zero-or-stale memory the caller must fill again. A page whose words are all zero may be
// reported dropped although it was not; that costs a read, never a wrong answer.
// All three take page-aligned spans that no other owner shares a page with.
bool vm_discard_supported();
bool vm_commit_discardable(void * p, size_t sz);
bool vm_offer(void * p, size_t sz);
bool vm_reclaim(void * p, size_t sz);
// Whether an offered span is still held, asked without taking it back: 1 held, 0 dropped, -1 when
// the platform cannot tell. Residency is the answer where an offered page stays mapped until the
// kernel takes it (mincore). Windows unmaps an offered range at once (out of the working set and
// PAGE_NOACCESS until reclaimed), so residency there reads "dropped" for every offer: -1.
int vm_offered_held(const void * p, size_t sz);

// Drop the resident pages of a FILE-BACKED mapping (the model's mmap) so a later touch refaults them
// from the file. Distinct from vm_evict, which decommits ANONYMOUS reserved-cache pages: this is for
// the clean, read-only weight mapping — MADV_DONTNEED discards the resident copy without writing
// anything back. Used after --dense-weights anon copies a dense tensor into its own anon buffer and
// rebinds onto it, to hand the now-unreferenced mmap pages back instead of leaving them double-
// resident until reclaim. A no-op where the platform cannot (Windows host: a file view is not
// VirtualFree-able, and the host build never streams). The range must be page-aligned by the caller.
void vm_drop_file_pages(void * p, size_t sz);

// Hand the system's cache of a FILE's pages back: the kernel keeps what was read through it (the
// readahead llama.cpp asks for over the whole mapping at load) until pressure takes it, and under a
// pressure with swap the kernel takes anonymous memory as well, the expert cache among it. Drops
// clean pages only; a page any process still maps stays. A no-op where the platform offers no such
// advice (Windows, Darwin: neither has posix_fadvise).
void drop_file_cache(const char * path);

// The same for one byte range of the file, whole pages inside it only. Where a range has just been
// unmapped (vm_drop_file_pages), this is what frees its pages now rather than when pressure finds
// them: unmapped, they are still the cache's and still count against a container's limit.
void drop_file_cache_range(const char * path, uint64_t offset, uint64_t size);

// Read the byte ranges {offset, size} of a file through the system's cache and discard the bytes:
// what brings those pages into the cache as a mapping's reads would, on every platform alike, with
// 64-bit offsets everywhere. In pieces of read_through_piece; `stop` (nullable) ends it between two.
// False when the file cannot be opened or a read fails.
bool read_through_cache(const char * path,
                        const std::vector<std::pair<uint64_t, uint64_t>> & ranges,
                        const std::atomic<bool> * stop = nullptr);
// Large enough that the reads run sequentially at the drive's rate, small beside any memory a model
// is run in.
inline constexpr size_t read_through_piece = size_t(8) << 20;

// Drop the resident pages of an ANONYMOUS range someone else allocated and still owns, keeping the
// range valid: the next touch maps a zero page instead of faulting. For memory whose contents moved
// elsewhere but whose allocation cannot be freed from here (a llama.cpp buffer a tensor no longer
// points at). Unlike vm_evict it never decommits, so a later write by the owner stays legal. A no-op
// on Windows (MEM_DECOMMIT would make that write fault; nothing moves memory away on the host build).
// The range must be page-aligned by the caller.
void vm_drop_anon_pages(void * p, size_t sz);

// Tell the kernel a FILE-BACKED range will be touched at random, so a fault brings in the page that
// faulted and nothing around it. The default is sequential readahead sized in the hundreds of KiB
// per fault, which is right for a weight that is read whole and wrong by two orders of magnitude for
// a table the graph gathers a few rows from per token: the neighbours are never used, and they
// occupy page cache the expert cache is competing for. A no-op on the Windows host (no streaming
// there). The range must be page-aligned by the caller.
void vm_advise_random(void * p, size_t sz);

// How many of a committed range's pages the kernel still has in RAM. Counts only the pages fully
// inside [p, p+sz) — the same clipping vm_evict's callers apply — and ADDS to *sampled/*resident,
// so several ranges can be summed into one fraction (the caller zeroes them first).
//
// This is the reclaim sensor. Nothing unprivileged can ask Android how much memory it will concede
// (MemAvailable counts our own file-backed weights as free; PSI is SELinux-blocked for apps), but a
// process may always ask about its OWN pages — and pages we wrote, then lost, are reclaim by
// definition. One mincore() serves a whole chunk of pages, so the cost is per range, not per page.
//
// Returns false when the platform cannot report residency (Windows host build), leaving the
// counters untouched: "unmeasured", which callers must not read as "nothing is resident".
bool vm_resident_sample(const void * p, size_t sz, size_t * sampled, size_t * resident);

// One file-backed VMA of this process: a virtual span [start, end) that maps the file starting at
// file_offset. The dense model weights are mmap'd, so this is how a file byte offset becomes an
// address to probe with mincore — addr(off) = start + (off - file_offset) for off in this VMA.
struct MappedRegion {
    uintptr_t start = 0;
    uintptr_t end = 0;
    uint64_t file_offset = 0;
};

// Read /proc/self/maps and collect every file-backed VMA whose pathname ends with `basename` — the
// model's file name, matched by suffix so it is found however the path is spelled. This is the
// self-inspection that IS allowed where /proc/vmstat was not: /proc/self/maps is the process's own,
// not a system file behind a vendor SELinux label. Returns false (out untouched) on the host build
// or if the file cannot be read — the caller reports that as unmeasured, never as "nothing resident".
bool file_mapped_regions(const char * basename, std::vector<MappedRegion> & out);

// Physical memory currently allocatable without paging, in bytes. 0 = unknown. Read once at init to
// size the expert cache to the device (--cache-mb auto); the budget is fixed for the run thereafter.
uint64_t mem_available_bytes();

// What a memory control group still lets this process take, in bytes: the tightest of its group
// and every ancestor it can see, each as the limit less what the group holds, plus the group's own
// file pages (active and inactive, as MemAvailable counts them), which the kernel drops before it
// would end anything. A container caps a
// process this way whatever the machine has free, and exceeding the cap ends the process, so it is
// the same kind of fact as a phone app's own share. 0 where no finite limit is set or none can be
// read (no cgroups, an unlimited group, another kernel). The paths are parameters so a test can
// stand a fake hierarchy up; production reads the defaults.
uint64_t cgroup_memory_left(const char * fs_root = "/sys/fs/cgroup", const char * self = "/proc/self/cgroup");

// The CPUs a control group's quota lets this process use, rounded up: the tightest quota over
// period of its group and every ancestor (v2 cpu.max, v1 cpu.cfs_quota_us / cpu.cfs_period_us). A
// container capped this way still shows every core of the host, and a short burst runs on all of
// them before the quota throttles it, so a probe of a few milliseconds measures a machine the
// sustained decode does not have. 0 where no quota is set or none can be read.
uint32_t cgroup_cpu_limit(const char * fs_root = "/sys/fs/cgroup", const char * self = "/proc/self/cgroup");

// Physical memory this machine has, 0 where it cannot be read. Not a sizing signal on its own -
// what a process may hold is `mem_available_bytes()` and what it may KEEP is a separate probe - but
// it bounds both, and it is what a rationale needs to say how much of the machine a plan is asking
// for. A phone with 12 GB total and 6 GB available is describing two different things.
uint64_t mem_total_bytes();

// Process-wide compute-decomposition counters, cumulative since process start; the caller deltas
// them across a single decode to split the per-token "compute" residual into its real causes.
// Both return 0 when the platform cannot report them, which the metrics treat as "unmeasured"
// rather than "zero work"; major_faults_counted() says which reading a 0 fault count is.
//
//   * major_faults(): hard page faults served from backing store (getrusage ru_majflt). A non-zero
//     per-token delta means a mmap-resident weight was re-faulted from flash *inside* the decode —
//     the >RAM residency stall that would otherwise masquerade as compute.
//   * process_cpu_seconds(): CPU time summed across all threads (CLOCK_PROCESS_CPUTIME_ID). Compared
//     against wall×threads it reveals occupancy: cpu≈wall×threads is genuine compute-bound work;
//     cpu≪wall×threads means the threads were descheduled or blocked (frequency cap, preemption,
//     fault wait) rather than computing.
uint64_t major_faults();
double process_cpu_seconds();
// Whether major_faults() counts anything here. Where it does not (no per-process count of HARD
// faults is published), a 0 is "unmeasured", and a report must not print it as "no faults".
bool major_faults_counted();

// Bytes one major fault moves: the page size. Faults are counted, but what a reader wants to know is
// how much memory the kernel took back and made us re-read — and a count only becomes that once it
// is multiplied by this. 194 MiB re-faulted in one token says what "47447 faults" does not.
size_t fault_bytes();

// Where this process's memory actually is, right now (/proc/self/status). The split is the whole
// point: the expert cache is ANONYMOUS, the model's weights are FILE-backed, and they are reclaimed
// by different mechanisms with different costs — anon goes to zram (swap), file pages are simply
// dropped. Watching them separately is how you tell "the kernel is taking my cache" apart from
// "the kernel is dropping the weights I mmap'd".
struct ProcessMemory {
    uint64_t rss_bytes = 0;      // VmRSS: everything resident
    uint64_t rss_anon_bytes = 0; // RssAnon: our cache lives here
    uint64_t rss_file_bytes = 0; // RssFile: the mmap'd model
    uint64_t swap_bytes = 0;     // VmSwap: anon we have already lost to zram

    // The process's own accounting, when published: its charged footprint and the bytes left
    // before its current limit. Neither can be reconstructed from RSS + swap. Known flags keep
    // an exhausted allowance (a valid zero) distinct from a platform that does not publish one.
    bool process_footprint_known = false;
    uint64_t process_footprint_bytes = 0;
    bool process_headroom_known = false;
    uint64_t process_headroom_bytes = 0;
};
// False when the platform cannot report (Windows host), leaving `out` untouched.
bool process_memory(ProcessMemory * out);

// What the device says about itself (/proc/meminfo). MemAvailable is the number that lies (it counts
// our own mmap'd weights as reclaimable), which is exactly why it is worth recording next to what we
// measure ourselves: the gap between them IS the story.
struct DeviceMemory {
    uint64_t available_bytes = 0; // MemAvailable
    uint64_t free_bytes = 0;      // MemFree
    uint64_t swap_free_bytes = 0; // SwapFree

    // What a reclaim could still free by compressing, as two facts. Each is 0 where the machine
    // does not publish it or the process may not read it, which is "will not say", not "none".
    //
    // The anonymous memory nobody has touched lately - the set compressed first. Where the
    // accounting does not split the inactive list by what backs it, this is that list less every
    // file-backed page: an under-count on purpose, since the available figure above already holds
    // the file-backed ones and a page counted in both would be promised twice.
    uint64_t idle_anon_bytes = 0;
    // What the in-RAM compressor holds, before and after: their ratio is this machine's own
    // measurement on its own data, not a figure for a compression algorithm in general.
    uint64_t compressed_stored_bytes = 0;
    uint64_t compressed_held_bytes = 0;
};
bool device_memory(DeviceMemory * out);

// What the kernel does with anonymous memory it wants back, as the facts that decide it. Each is
// read from the kernel's own accounting - a syscall or a counter, never a benchmark - so asking
// costs microseconds. False when nothing could be read at all.
struct ReclaimFacts {
    // What is left of this process's OWN share of memory, where the kernel keeps one and ends the
    // process for exceeding it; 0 where no such share is published (the limit is the machine's).
    uint64_t own_share_left = 0;
    bool compresses = false; // anonymous memory is compressed in RAM before anything else happens to it
    bool swaps = false;      // a backing store on disk exists for anonymous memory
};
bool reclaim_facts(ReclaimFacts * out);

} // namespace bmoe::pio
