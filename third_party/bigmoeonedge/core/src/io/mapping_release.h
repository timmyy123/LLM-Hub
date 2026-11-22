#pragma once
// Release a model file's memory mapping once nothing reads through it any more.
//
// Why this exists. llama.cpp maps every gguf it loads (use_mmap is load-bearing for the streamer:
// the native layout is what the expert rebind points into) and keeps the mapping for the model's
// lifetime. On Windows that mapping has a cost the streamer pays on every read: while a section of
// the file is alive, NTFS serialises concurrent unbuffered (FILE_FLAG_NO_BUFFERING) reads on that
// file. Measured with tools/bmoe-iobench on the same drive and request shape (576 KiB, 4 lanes):
// 2400-2660 MiB/s with no mapping, 895-930 with one, i.e. four lanes deliver exactly one lane's
// throughput. Unmapping the view alone does not help; the section must be closed too.
//
// Under `--dense-weights anon`/`pinned` with every file tensor accounted for (streamed experts,
// copied dense weights, row-streamed tables), nothing dereferences the mapping after load, so it
// can go. llama.cpp offers no public API to drop it (its Windows unmap_fragment is a no-op), so
// this module finds the process's own views and section handles of the file and releases them.
//
// What it does NOT do: touch llama.cpp. The model still believes it owns the mapping and will
// UnmapViewOfFile/CloseHandle it at free time. Both are made harmless here rather than left to
// luck: the view's address range is re-occupied by a view of an empty, uncommitted pagefile section
// (so no later allocation can land there, and the model's unmap succeeds by releasing that
// placeholder instead of failing with a warning), and the closed handle's slot is re-occupied by
// an inert event (so a handle allocated later cannot inherit the value and be closed by mistake).
// The placeholders are returned to the caller, who releases them AFTER the model is freed.
//
// On POSIX the same release is done with munmap over the file's VMAs from /proc/self/maps, with
// anonymous placeholders in their place. Android (f2fs) was measured not to serialise direct reads
// against a live mapping, so there it buys nothing; it exists so that claim can be re-measured
// rather than assumed on the next platform.
#include <cstdint>
#include <string>
#include <vector>

namespace bmoe::pio {

struct MappingReleaseReport {
    bool supported = false; // false: nothing to do on this platform (POSIX); the rest is zero
    int views_unmapped = 0;
    int sections_closed = 0;
    uint64_t bytes = 0;   // total size of the views released
    int plugs_missed = 0; // closed sections whose handle slot could not be re-occupied
    std::string error;    // first hard failure, empty on success; partial work is still reported
};

// Everything a release leaves behind on purpose, to be released only after the model owning the
// original mapping has been freed (see the header comment). Movable, not copyable; releasing twice
// is a no-op.
class MappingPlaceholders {
public:
    MappingPlaceholders() = default;
    ~MappingPlaceholders();
    MappingPlaceholders(MappingPlaceholders &&) noexcept;
    MappingPlaceholders & operator=(MappingPlaceholders &&) noexcept;
    MappingPlaceholders(const MappingPlaceholders &) = delete;
    MappingPlaceholders & operator=(const MappingPlaceholders &) = delete;

    void release();
    bool empty() const { return reserved_.empty() && viewed_.empty() && plugs_.empty(); }

private:
    friend MappingReleaseReport release_file_mappings(const std::vector<std::string> &, MappingPlaceholders *);
    std::vector<std::pair<void *, size_t>> reserved_; // address ranges held in place of the views
    std::vector<std::pair<void *, size_t>> viewed_;   // Windows: placeholder views the model unmaps
    std::vector<void *> plugs_;                       // inert handles occupying closed sections' slots
};

// Release every view and section this process holds on the files in `paths` (UTF-8). Files that
// cannot be opened for identification are skipped and named in `report.error`. `out` receives the
// placeholders; it must outlive the model.
MappingReleaseReport release_file_mappings(const std::vector<std::string> & paths, MappingPlaceholders * out);

// How many of `addresses` still point inside a mapping of one of `paths`. This is what makes a
// release decidable rather than guessed: releasing is correct only while nothing dereferences the
// mapping, and the caller can ask about every pointer it knows instead of reasoning about which
// tensor names ought to have been rebound. It answers for the pointers it is given and for no
// others, so a caller that cannot enumerate everything still has to keep the flag opt-in.
size_t addresses_in_file_mappings(const std::vector<std::string> & paths, const std::vector<const void *> & addresses);

} // namespace bmoe::pio
