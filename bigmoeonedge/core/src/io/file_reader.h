// A pooled, positioned file reader with optional cache-bypassing I/O (O_DIRECT on POSIX,
// FILE_FLAG_NO_BUFFERING on Windows, F_NOCACHE on Apple).
//
// One reader owns N lane fds and N bounce buffers, so N threads can read distinct byte ranges of the
// same file concurrently without contending. Where the platform's direct mode is an I/O mode that
// rejects unaligned access (O_DIRECT, NO_BUFFERING), a direct read aligns its window to the read
// alignment, pulls it into the lane's bounce buffer, and memcpy's the requested interior out — the
// mechanics those modes require. Apple's F_NOCACHE is not such a mode — a caching hint on the
// descriptor — so a direct reader there keeps ordinary pread semantics and skips the bounce
// entirely; see pio::direct_needs_alignment(). `direct` is a property of the reader, chosen by
// whoever opens it: the expert streamer and the dense-weights loader each construct their own, so
// one can bypass the page cache while the other does not — the two decisions are independent, not
// a shared global.
//
// The O_DIRECT request is verified once at open: on storage where a direct read returns garbage
// (some FUSE-backed emulated volumes) the reader silently falls back to buffered I/O for the file,
// so a caller never has to reason about the storage — it asks for direct and gets correct bytes.
// direct() reports the effective mode after all of that, which is what telemetry must print, not
// what the caller requested.
#pragma once

#include "platform_io.h"

#include <atomic>
#include <string>
#include <vector>

namespace bmoe {

class FileReader {
public:
    FileReader() = default;
    ~FileReader();

    FileReader(const FileReader &) = delete;
    FileReader & operator=(const FileReader &) = delete;

    // Open `path` with `lanes` independent primary fds (+ a buffered fd per lane for the sub-alignment
    // EOF tail an alignment-constrained direct read cannot cover) and an aligned bounce per lane sized
    // for a read of up to `max_read` bytes (a larger read grows its lane's bounce). `direct` requests
    // cache bypass; it is verified and silently downgraded where the platform or the storage refuses
    // or mis-serves it — direct() then reports the effective mode. Reads align to the granule the
    // platform states for this file (pio::direct_alignment, asked of the descriptor just opened)
    // where the platform's direct mode demands it. Returns false on any open/alloc failure.
    bool open(const std::string & path, int lanes, bool direct, size_t max_read);
    void close();
    // Close every fd and open the same file again with the same lanes and bounce size; the alignment
    // is asked again of the new descriptor.
    // Exists for one reason: on Windows an unbuffered handle opened while the file had a live
    // section keeps serialising against it even after the section is gone, so once the model's
    // mapping is released the lanes must be reopened to read at the drive's rate (see
    // mapping_release.h). Not thread-safe: no read may be in flight on any lane. Accounting is kept.
    bool reopen();

    bool is_open() const { return !fds_.empty(); }
    bool direct() const { return direct_; } // the effective cache-bypass mode, not the request
    uint64_t file_size() const { return fsize_; }
    int lanes() const { return (int) fds_.size(); }
    // The granule this reader's direct windows are cut to: what the platform stated for the file, or
    // pio::direct_align_fallback where it said nothing - align_reported() tells the two apart, and
    // only a reported value is a fact about the storage.
    size_t align() const { return align_; }
    bool align_reported() const { return align_reported_; }
    // The alignment a caller's DESTINATION buffer for these reads should have: the read granule, and
    // never less than a page, so a buffer the vm_* calls later clip to pages holds whole pages.
    size_t buffer_align() const;

    // Read `nbytes` at file offset `off` into `dst`, on `lane` (0 <= lane < lanes()). Thread-safe
    // across distinct lanes — each has its own fd and bounce. Returns what was actually pulled from
    // the drive — the aligned window (>= nbytes) where the platform's direct mode requires one,
    // exactly the requested bytes otherwise — which is what the bandwidth must be judged against;
    // or -1 on I/O error or on a range that runs past the end of the file (all of `nbytes` is
    // written, or the read fails). A zero-length read is a no-op returning 0. The lane's bounce grows
    // if a windowed read needs more.
    long long read(int lane, void * dst, uint64_t off, uint64_t nbytes);

    // Aggregate accounting since open, summed across lanes.
    long long read_bytes() const { return read_bytes_.load(std::memory_order_relaxed); }
    long long syscall_ns() const { return syscall_ns_.load(std::memory_order_relaxed); }

private:
    std::vector<pio::fd_t> fds_;     // primary (cache-bypassing where achieved) per lane
    std::vector<pio::fd_t> fds_buf_; // buffered fallback per lane, for the sub-alignment EOF tail
    std::vector<void *> bounces_;
    std::vector<size_t> bounce_sz_;
    size_t align_ = pio::direct_align_fallback;
    bool align_reported_ = false;
    uint64_t fsize_ = 0;
    bool direct_ = false;        // cache bypass actually in effect for the fds (uncached descriptors),
                                 // after the open's own report and every fallback
    bool aligned_reads_ = false; // direct_ AND the platform's direct mode rejects unaligned reads —
                                 // gates the read mechanics: window rounding, bounce, buffered tail fd
    // What open() was asked for, so reopen() can ask for the same thing again. direct_ above is the
    // outcome, which a reopen must be free to reach on its own rather than inherit.
    std::string path_;
    int lanes_ = 0;
    bool direct_req_ = false;
    size_t max_read_ = 0;
    std::atomic<long long> read_bytes_{0};
    std::atomic<long long> syscall_ns_{0};
};

} // namespace bmoe
