// The measured half of the machine profile: what this storage actually does with the reads this
// engine actually issues.
//
// Two facts come out of here, and neither can be inferred from a platform name:
//
//   * The rate curve. A 4 KiB read returns 2% of peak on one desktop SSD and 6.8% on a phone's
//     UFS, and lanes scale on one where they saturate on the other. The only honest source of a
//     lane count is therefore this machine, at this model's expert slice size.
//
//   * Whether a live mapping of the model serialises concurrent uncached reads. It is true on one
//     desktop filesystem and false on a phone's, and where it is true it costs most of the read
//     path's throughput. Measured by doing exactly that: read with a mapping alive, drop the
//     mapping, reopen the lanes, read again.
//
// The probe is bounded in bytes, not in time, and it is not small. The decisive point - the model's
// own slice - moves 512 MiB per sample, three repeats at each of three lane counts: 4.5 GiB. The
// rest is counted in requests: the curve's two neighbouring sizes and both arms of the mapping
// question come to about 2150 slices' worth, and the refault price to 480 pages (2 MiB at 4 KiB). At
// a 512 KiB slice that is roughly 5.5 GiB read from the file in all. It runs only when asked.

#include "bmoe/probe.h"

#include "../io/file_reader.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace bmoe {

namespace {

using clock_t_ = std::chrono::steady_clock;

// The lane counts the curve is drawn at: one, and doublings up to four. Where lanes stop paying is
// the thing being looked for, and doubling brackets that knee between two points instead of paying
// for every count on the way; the storage this was measured on had stopped gaining by four. The
// mapping question is asked at the largest, where concurrent uncached reads have the most to
// serialise.
constexpr int k_lane_counts[] = {1, 2, 4};
constexpr int k_lanes_max = k_lane_counts[sizeof(k_lane_counts) / sizeof(k_lane_counts[0]) - 1];

// Samples per point that decides something, and per side of the mapping question: the fewest whose
// median a single outlier cannot move.
constexpr int k_repeats = 3;

// A live mapping of the model file, held only for as long as the probe needs to read against it.
// This is the one thing the probe cannot ask the OS about and has to reproduce.
class ScopedMapping {
public:
    explicit ScopedMapping(const char * path) { open(path); }
    ~ScopedMapping() { close(); }
    ScopedMapping(const ScopedMapping &) = delete;
    ScopedMapping & operator=(const ScopedMapping &) = delete;

    bool ok() const { return ok_; }
    void close();

    // Make part of the mapping resident, the way a loaded model's mapping is where the engine has
    // been reading. The pathology being reproduced is an interaction between a WARM mapping and
    // concurrent uncached reads; a mapping that has never been touched holds no pages and cannot
    // interact with anything. Bounded on purpose - the file is larger than memory, and the point is
    // to look like a working set, not to fault the whole model in.
    void warm(uint64_t bytes);

private:
    void open(const char * path);
    bool ok_ = false;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE section_ = nullptr;
    void * view_ = nullptr;
#else
    int fd_ = -1;
    void * addr_ = nullptr;
    size_t len_ = 0;
#endif
};

#if defined(_WIN32)
void ScopedMapping::open(const char * path) {
    int wn = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
    if (wn <= 0) return;
    std::vector<wchar_t> w((size_t) wn);
    MultiByteToWideChar(CP_UTF8, 0, path, -1, w.data(), wn);
    file_ = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) return;
    section_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!section_) return;
    // Map the WHOLE file, the way a model loader does. A token-sized view is not equivalent: the
    // measured pathology tracks how much of the file is actually mapped and resident, so a probe
    // that maps a megabyte reproduces nothing and would report a confident "no".
    view_ = MapViewOfFile(section_, FILE_MAP_READ, 0, 0, 0);
    ok_ = view_ != nullptr;
}

void ScopedMapping::close() {
    if (view_) {
        UnmapViewOfFile(view_);
        view_ = nullptr;
    }
    if (section_) {
        CloseHandle(section_);
        section_ = nullptr;
    }
    if (file_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
    }
    ok_ = false;
}

void ScopedMapping::warm(uint64_t bytes) {
    if (!ok_ || !view_) return;
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(view_, &mbi, sizeof(mbi)) == 0) return;
    const uint64_t len = (uint64_t) mbi.RegionSize;
    const uint64_t want = std::min<uint64_t>(bytes, len);
    volatile const char * p = (volatile const char *) view_;
    volatile char sink = 0;
    const uint64_t page = (uint64_t) pio::vm_page(); // one touch per page faults the whole span in
    for (uint64_t off = 0; off < want; off += page)
        sink = p[off];
    (void) sink;
}
#else
void ScopedMapping::open(const char * path) {
    fd_ = ::open(path, O_RDONLY);
    if (fd_ < 0) return;
    struct stat st {};
    if (::fstat(fd_, &st) != 0 || st.st_size <= 0) return;
    len_ = (size_t) st.st_size;
    addr_ = ::mmap(nullptr, len_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (addr_ == MAP_FAILED) {
        addr_ = nullptr;
        return;
    }
    ok_ = true;
}

void ScopedMapping::close() {
    if (addr_) {
        ::munmap(addr_, len_);
        addr_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    ok_ = false;
}

void ScopedMapping::warm(uint64_t bytes) {
    if (!ok_ || !addr_) return;
    const uint64_t want = std::min<uint64_t>(bytes, (uint64_t) len_);
    volatile const char * p = (volatile const char *) addr_;
    volatile char sink = 0;
    const uint64_t page = (uint64_t) pio::vm_page(); // one touch per page faults the whole span in
    for (uint64_t off = 0; off < want; off += page)
        sink = p[off];
    (void) sink;
}
#endif

// How the offsets are drawn. The distinction matters because the two questions this probe answers
// are not the same question. What a lane count is worth is a property of the storage, and spreading
// the reads over the whole file is the cleanest way to ask it. Whether a live mapping serialises
// our reads is a property of how the ENGINE reads: it walks a layer's expert slices, which are
// clustered in the file and advance through it as the token progresses. A uniform draw over 21 GiB
// reproduces none of that locality, and the probe that used one missed a 24% effect the engine sees.
enum class Pattern {
    Spread,   // uniform over the file: asks the storage what it can do
    Clustered // a window of neighbouring slices that advances: asks it what happens to US
};

// Read `per_lane` blocks of `block` bytes per lane and return the aggregate rate in MiB/s. Offsets
// are drawn from a stream seeded by the caller so two points of the curve do not read the same bytes
// and warm each other's drive cache.
double
timed_read(FileReader & rd, uint64_t block, int lanes, int per_lane, uint64_t seed, Pattern pat = Pattern::Spread) {
    const uint64_t fsize = rd.file_size();
    if (fsize <= block * 4) return 0.0;
    const uint64_t span = fsize - block;
    const uint64_t granule = (uint64_t) rd.align();

    std::vector<std::vector<char>> bufs((size_t) lanes);
    for (auto & b : bufs)
        b.resize((size_t) block);

    const auto t0 = clock_t_::now();
    std::vector<std::thread> ths;
    std::vector<long long> got((size_t) lanes, 0);
    ths.reserve((size_t) lanes);
    for (int l = 0; l < lanes; ++l) {
        ths.emplace_back([&, l] {
            uint64_t x = seed + (uint64_t) l * 0x9E3779B97F4A7C15ull;
            long long total = 0;
            // The clustered walk: a window a layer's worth of slices wide, drawn inside, advancing
            // once per round so the whole read sweeps the file the way a generated token does.
            const uint64_t window = std::min<uint64_t>(span, block * 64);
            for (int i = 0; i < per_lane; ++i) {
                // xorshift64*, so the offsets are spread without pulling in <random>
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                const uint64_t draw = x * 0x2545F4914F6CDD1Dull;
                uint64_t off = draw % span;
                if (pat == Pattern::Clustered) {
                    const uint64_t base = (span / (uint64_t) per_lane) * (uint64_t) i;
                    off = (base + draw % window) % span;
                }
                off -= off % granule; // whole read granules, so no request pays a partial-block over-read
                const long long n = rd.read(l, bufs[(size_t) l].data(), off, block);
                if (n > 0) total += n;
            }
            got[(size_t) l] = total;
        });
    }
    for (auto & t : ths)
        t.join();
    const double secs = std::chrono::duration<double>(clock_t_::now() - t0).count();
    if (secs <= 0.0) return 0.0;

    long long bytes = 0;
    for (long long g : got)
        bytes += g;
    return (double) bytes / (1024.0 * 1024.0) / secs;
}

// The median of `repeats` measurements of the same point. A drive's answer to the same question
// varies run to run, and the lane decision is a comparison between points a few percent apart: on
// one desktop SSD a single sample flipped the answer between one lane and two across runs, which is
// the probe reporting its own noise. The median is the cheapest estimator that a single outlier
// cannot move, and it costs only the repeats of the one point that decides something.
double median_rate(FileReader & rd,
                   uint64_t block,
                   int lanes,
                   int per_lane,
                   uint64_t seed,
                   int repeats,
                   Pattern pat = Pattern::Spread,
                   double * spread = nullptr) {
    if (repeats <= 1) return timed_read(rd, block, lanes, per_lane, seed, pat);
    std::vector<double> r;
    r.reserve((size_t) repeats);
    for (int i = 0; i < repeats; ++i)
        r.push_back(timed_read(rd, block, lanes, per_lane, seed + (uint64_t) i * 0x9E3779B9ull, pat));
    std::sort(r.begin(), r.end());
    // The spread across repeats, as a ratio. A probe that does not report its own noise invites a
    // plan to quote a rate to three digits that its next run will contradict.
    if (spread && r.front() > 0.0) *spread = r.back() / r.front();
    return r[r.size() / 2];
}

// How many requests each lane must issue for one sample to last long enough to mean anything. A
// fixed request count makes the decisive sample as short as the storage is fast - 160 requests of
// 664 KiB is 106 MiB, which a desktop NVMe finishes in 40 ms - and a 40 ms measurement on a
// general-purpose OS reports scheduling noise. Measured: across three consecutive runs the same
// machine answered 1116, 1526 and 2583 MiB/s at the same point, and the plan's predicted ms/token
// moved with it by more than 2x. Sizing by BYTES instead makes the sample's duration a property of
// the measurement rather than of the device.
int requests_per_lane(uint64_t block, int lanes, uint64_t target_bytes, int floor_count) {
    if (block == 0 || lanes <= 0) return floor_count;
    const uint64_t per = target_bytes / (block * (uint64_t) lanes);
    return (int) std::max<uint64_t>((uint64_t) floor_count, per);
}

} // namespace

void probe_storage(HardwareProfile & hw, const char * model_path, uint64_t slice_bytes) {
    if (!model_path || !*model_path) return;

    const uint64_t slice = slice_bytes ? slice_bytes : (512u << 10);
    const int per_lane = 40; // per point on the curve's shape: one short sample says enough

    // Bytes ONE sample must move at the point that decides something. Sized for duration rather
    // than for a request count, so the measurement does not get shorter as the storage gets faster.
    const uint64_t decisive_bytes = 512ull << 20;

    // The largest read the curve issues: one point above the model's slice, where lanes stop paying.
    const uint64_t largest = slice * 4;

    // Measure with a live mapping first, then release it and reopen the lanes. The order is the one
    // that matters on the storage where this bites: a handle opened while a section was alive keeps
    // serialising against it even after the section is gone, so the mapped case has to come first
    // and the unmapped case has to reopen.
    double mapped_rate = 0.0;
    {
        ScopedMapping map(model_path);
        if (map.ok()) {
            FileReader rd;
            if (rd.open(model_path, k_lanes_max, /*direct=*/true, (size_t) slice)) {
                map.warm(slice * 64); // a loaded model's mapping is resident where it has been read
                mapped_rate = median_rate(rd, slice, k_lanes_max, per_lane, 0xC0FFEEull, k_repeats, Pattern::Clustered);
            }
        }
    }

    FileReader rd;
    if (!rd.open(model_path, k_lanes_max, /*direct=*/true, (size_t) largest)) return;
    // What the file's filesystem states, or nothing: the reader's fallback granule is a substitute
    // it reads with, not a fact about this storage.
    hw.storage.align = rd.align_reported() ? (uint32_t) rd.align() : 0;
    hw.storage.direct_ok = rd.direct() ? Tri::Yes : Tri::No;

    // The sizes worth knowing are the ones around the read this engine issues. Below the model's
    // own slice is where a storage class shows whether it punishes small requests; above it is
    // where lanes stop paying. Never below one read granule, the smallest request the storage serves.
    const uint64_t sizes[3] = {std::max<uint64_t>((uint64_t) rd.align(), slice / 4), slice, largest};

    uint64_t seed = 0x1234567ull;
    for (uint64_t sz : sizes) {
        for (int lanes : k_lane_counts) {
            // Only the model's own slice size decides anything - it is where the lane count is read
            // off and where the expert lane is priced - so it is the only size worth paying repeats
            // AND a long sample for. The neighbours are there to show the shape of the curve, and
            // one short sample says enough about a shape.
            const bool decisive = sz == slice;
            const int repeats = decisive ? k_repeats : 1;
            const int reqs = decisive ? requests_per_lane(sz, lanes, decisive_bytes, per_lane) : per_lane;
            double spread = 0.0;
            const double r = median_rate(rd, sz, lanes, reqs, seed, repeats, Pattern::Spread, &spread);
            if (decisive && spread > hw.storage.rate_spread) hw.storage.rate_spread = spread;
            seed += 0x9E3779B9ull;
            if (r <= 0.0) continue;
            hw.storage.rate_curve.push_back({(uint32_t) sz, (uint32_t) lanes, r});
        }
    }

    // The price of a REFAULT, measured as the kernel pays it: one page at a time - this machine's
    // page, which is 16 KiB on some and 4 KiB on others - scattered, one lane. This is what a group left to the file
    // mapping costs per byte once pressure has dropped its pages, and it is the term that decides whether residency is
    // worth buying at all. It is deliberately not derived as a fraction of the sequential rate: on the test host it is
    // 23 MiB/s against 2609 sequential, 116x, and a plan that assumed a fraction would under-spend on residency by two
    // orders of magnitude.
    hw.storage.refault_mibs =
        median_rate(rd, (uint64_t) pio::vm_page(), 1, per_lane * 4, 0x5EEDFA11ull, k_repeats, Pattern::Spread);

    // The comparable point for the mapping question, read the way the engine reads rather than the
    // way the curve is drawn: same size, same lanes, same pattern as the mapped arm above.
    const double unmapped_rate =
        median_rate(rd, slice, k_lanes_max, per_lane, 0xC0FFEEull, k_repeats, Pattern::Clustered);

    // The verdict, from the two comparable points. The threshold is deliberately loose: what is
    // being detected is a collapse to roughly one lane's throughput, not a few percent of noise.
    hw.storage.rate_mapped_mibs = mapped_rate;
    hw.storage.rate_unmapped_mibs = unmapped_rate;

    // This probe reports Yes or nothing, never No.
    //
    // It used to draw both arms uniformly at random over the whole file, and on a desktop where the
    // ENGINE gains 24% of decode from releasing the mapping - its own read rate goes 871 -> 1680
    // MiB/s - it saw the two arms within 2% of each other. Two things it was not reproducing have
    // since been fixed: the mapping is now warmed before the mapped arm reads, because an untouched
    // mapping holds no pages and cannot interfere with anything, and both arms now walk clustered,
    // advancing windows the way the streamer walks a layer's expert slices.
    //
    // The fix worked where the old probe failed: on that same desktop it now reads 1151 against 2806
    // MiB/s across repeated runs, an effect of the right size and sign, where before it saw the two
    // arms within 2% of each other.
    //
    // The asymmetry stays anyway. What has been established is that this instrument can see the
    // effect where it exists; nothing yet establishes that its silence means absence, because no
    // machine with a KNOWN negative has been put in front of it. Reporting a No as `measured` on
    // that basis would be the confident wrong answer this whole design exists to avoid.
    if (mapped_rate > 0.0 && unmapped_rate > 0.0 && mapped_rate < 0.7 * unmapped_rate) {
        hw.storage.mapping_serialises_reads = Tri::Yes;
    }
}

} // namespace bmoe
