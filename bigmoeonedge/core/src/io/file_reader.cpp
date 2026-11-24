#include "file_reader.h"
#include "bmoe/log.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace bmoe {

using clock_t_ = std::chrono::steady_clock;

FileReader::~FileReader() {
    close();
}

bool FileReader::open(const std::string & path, int lanes, bool direct, size_t max_read) {
    if (is_open()) return false;
    // Recorded after the guard: a refused call must not overwrite what reopen() replays.
    path_ = path;
    lanes_ = lanes;
    direct_req_ = direct;
    max_read_ = max_read;
    direct_ = direct;
    const int n = lanes < 1 ? 1 : lanes;

    pio::fd_t primary = pio::open_read(path.c_str(), direct, &direct_); // direct_ := the open's real outcome
    if (!pio::fd_ok(primary) && direct) { // the platform refused the direct open outright — try buffered
        direct_ = false;
        primary = pio::open_read(path.c_str(), false);
    }
    if (!pio::fd_ok(primary)) {
        log_printf(LogLevel::Error, "open %s failed", path.c_str());
        return false;
    }
    fsize_ = pio::file_size(primary);
    // Asked of the descriptor the reads will use, so the granule is the one for the filesystem this
    // path is actually on. Asked even when the read is buffered: it is also what the storage probe
    // reports, and the question costs one stat.
    const size_t stated = pio::direct_alignment(primary);
    align_reported_ = stated != 0;
    align_ = align_reported_ ? stated : pio::direct_align_fallback;
    // A window starts up to one granule before the read and ends up to one granule after it.
    const size_t bounce_cap = max_read + 2 * align_;

    // The read mechanics follow the platform's direct MODE, not the direct FACT: O_DIRECT and
    // FILE_FLAG_NO_BUFFERING reject unaligned windows and need the bounce and the buffered tail fd;
    // Apple's F_NOCACHE is a caching hint, so an uncached reader there keeps plain preads.
    aligned_reads_ = direct_ && pio::direct_needs_alignment();

    // Verify O_DIRECT actually returns correct bytes on this storage. On some emulated / FUSE-backed
    // volumes (e.g. an app-private dir under /storage/emulated) the open SUCCEEDS but direct reads
    // return garbage, silently corrupting weights → nonsense output. Compare one aligned block read
    // directly against the same block read buffered; on any mismatch, fall back to buffered I/O for
    // this file. On real filesystems (adb-pushed models, desktop) the two match and O_DIRECT is kept.
    if (aligned_reads_ && fsize_ >= (uint64_t) align_) {
        uint64_t voff = (fsize_ / 2) & ~(uint64_t) (align_ - 1);
        if (voff + align_ > fsize_) voff = 0;
        void * dbuf = pio::alloc_aligned(align_, align_);
        pio::fd_t vfd = pio::open_read(path.c_str(), false);
        if (dbuf && pio::fd_ok(vfd)) {
            std::vector<char> bbuf(align_);
            const long long want = (long long) align_;
            const long long gd = pio::pread_at(primary, dbuf, align_, voff);
            const long long gb = pio::pread_at(vfd, bbuf.data(), align_, voff);
            const bool bad = gd != want || gb != want || std::memcmp(dbuf, bbuf.data(), align_) != 0;
            if (bad) {
                log_printf(LogLevel::Warn, "O_DIRECT returns wrong data on this storage — using buffered I/O");
                direct_ = false;
                pio::close_fd(primary);
                primary = pio::open_read(path.c_str(), false);
            }
        }
        if (dbuf) pio::aligned_free(dbuf);
        if (pio::fd_ok(vfd)) pio::close_fd(vfd);
        if (!pio::fd_ok(primary)) {
            log_printf(LogLevel::Error, "reopen after O_DIRECT check failed");
            return false;
        }
        aligned_reads_ = aligned_reads_ && direct_; // a verify downgrade reverts the mechanics too
    }

    // A private fd + bounce per lane so concurrent reads never contend. Lane 0 inherits the fd already
    // opened (and verified) above; the rest open their own.
    fds_.assign(n, pio::fd_invalid);
    fds_buf_.assign(n, pio::fd_invalid);
    bounces_.assign(n, nullptr);
    bounce_sz_.assign(n, 0);
    for (int lane = 0; lane < n; ++lane) {
        fds_[lane] = (lane == 0) ? primary : pio::open_read(path.c_str(), direct_);
        fds_buf_[lane] = aligned_reads_ ? pio::open_read(path.c_str(), false) : pio::fd_invalid;
        if (!pio::fd_ok(fds_[lane])) {
            log_printf(LogLevel::Error, "lane %d open failed", lane);
            close();
            return false;
        }
        // Not fatal — the buffered fd is only needed for a read that runs into a sub-alignment EOF
        // tail, which expert slices never do. But say so here, where the cause (fd pressure) is still
        // visible, rather than leaving a later tail read to fail with an unexplained EINVAL.
        if (aligned_reads_ && !pio::fd_ok(fds_buf_[lane]))
            log_printf(LogLevel::Warn, "lane %d buffered tail fd unavailable — EOF-tail reads will fail", lane);
        // No bounce where reads stay plain (Apple F_NOCACHE): a per-lane aligned reservation next to
        // the expert cache would buy nothing there.
        bounces_[lane] = aligned_reads_ ? pio::alloc_aligned(align_, bounce_cap) : nullptr;
        if (aligned_reads_ && !bounces_[lane]) {
            log_printf(LogLevel::Error, "lane %d bounce alloc failed", lane);
            close();
            return false;
        }
        bounce_sz_[lane] = aligned_reads_ ? bounce_cap : 0;
    }
    return true;
}

long long FileReader::read(int lane, void * dst, uint64_t off, uint64_t nbytes) {
    if (nbytes == 0) return 0;
    const pio::fd_t fd = fds_[lane];

    // Every caller asks for bytes a gguf header says the file holds. A range that ends past the end
    // of the file is a header pointing outside it (a truncated download, a damaged shard): an error,
    // never a short read the caller would take for weights, nor a window padded with whatever the
    // bounce held before. Written so off + nbytes cannot overflow.
    if (fsize_ && (off > fsize_ || nbytes > fsize_ - off)) {
        log_printf(LogLevel::Error, "read of %llu bytes at %llu runs past the end of %s (%llu bytes)",
                   (unsigned long long) nbytes, (unsigned long long) off, path_.c_str(), (unsigned long long) fsize_);
        return -1;
    }

    // Plain reads: no alignment constraint, so the bounce buys nothing. Read straight into the
    // caller's memory. This is the path for any reader whose direct mode imposes no alignment — the
    // fallback after a refused open or a failed open-time verify, and Apple's F_NOCACHE, which is
    // uncached without being an I/O mode. It must not also pay an extra copy of every byte plus a
    // leading partial-block over-read just to share the mechanics O_DIRECT needs.
    if (!aligned_reads_) {
        const uint64_t end = off + nbytes; // inside the file: checked above
        const auto t0 = clock_t_::now();
        for (uint64_t a = off; a < end;) {
            long long got = pio::pread_at(fd, (char *) dst + (a - off), (size_t) (end - a), a);
            if (got <= 0) {
                log_printf(LogLevel::Error, "pread failed at %llu", (unsigned long long) a);
                return -1;
            }
            a += (uint64_t) got;
        }
        const auto t1 = clock_t_::now();
        syscall_ns_.fetch_add((long long) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        const long long moved = (long long) (end - off);
        read_bytes_.fetch_add(moved);
        return moved; // no aligned window here: what was asked for is what was pulled
    }

    const uint64_t a0 = off & ~(uint64_t) (align_ - 1);
    const uint64_t a1 = (off + nbytes + align_ - 1) & ~(uint64_t) (align_ - 1);
    const size_t len = (size_t) (a1 - a0);
    if (bounce_sz_[lane] < len) {
        // Clear the size BEFORE the allocation: on failure the lane must look empty, not keep
        // advertising the freed buffer's capacity — a later smaller read would skip the realloc and
        // pread into a null pointer, turning one transient OOM into a permanently dead lane.
        if (bounces_[lane]) pio::aligned_free(bounces_[lane]);
        bounces_[lane] = nullptr;
        bounce_sz_[lane] = 0;
        bounces_[lane] = pio::alloc_aligned(align_, len);
        if (!bounces_[lane]) {
            log_printf(LogLevel::Error, "bounce realloc %zu failed", len);
            return -1;
        }
        bounce_sz_[lane] = len;
    }
    char * b = (char *) bounces_[lane];
    const pio::fd_t fd_buf = fds_buf_[lane];
    // The aligned window may round past the end of the file; the request itself does not.
    const uint64_t read_end = (fsize_ && a1 > fsize_) ? fsize_ : a1;
    const uint64_t bulk_end = read_end & ~(uint64_t) (align_ - 1);

    const auto t0 = clock_t_::now();
    for (uint64_t a = a0; a < bulk_end;) {
        long long got = pio::pread_at(fd, b + (a - a0), (size_t) (bulk_end - a), a);
        if (got <= 0) {
            log_printf(LogLevel::Error, "pread failed at %llu", (unsigned long long) a);
            return -1;
        }
        a += (uint64_t) got;
    }
    if (bulk_end < read_end && !pio::fd_ok(fd_buf)) {
        // The O_DIRECT fd would reject the sub-alignment length outright; say what is actually wrong.
        log_printf(LogLevel::Error, "EOF tail at %llu needs the buffered fd, which failed to open",
                   (unsigned long long) bulk_end);
        return -1;
    }
    for (uint64_t a = bulk_end; a < read_end;) { // sub-alignment EOF tail via the buffered fd
        long long got = pio::pread_at(fd_buf, b + (a - a0), (size_t) (read_end - a), a);
        if (got <= 0) {
            log_printf(LogLevel::Error, "tail pread failed at %llu", (unsigned long long) a);
            return -1;
        }
        a += (uint64_t) got;
    }
    const auto t1 = clock_t_::now();
    syscall_ns_.fetch_add((long long) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    std::memcpy(dst, b + (off - a0), (size_t) nbytes);
    const long long window = (long long) (read_end - a0);
    read_bytes_.fetch_add(window);
    return window; // the aligned window pulled — what the effective bandwidth is judged against
}

bool FileReader::reopen() {
    if (!is_open()) return false;
    const std::string path = path_;
    const int lanes = lanes_;
    const bool direct = direct_req_;
    const size_t max_read = max_read_;
    close();
    return open(path, lanes, direct, max_read);
}

size_t FileReader::buffer_align() const {
    const size_t page = pio::vm_page();
    return align_ > page ? align_ : page;
}

void FileReader::close() {
    for (pio::fd_t fd : fds_)
        if (pio::fd_ok(fd)) pio::close_fd(fd);
    for (pio::fd_t fd : fds_buf_)
        if (pio::fd_ok(fd)) pio::close_fd(fd);
    for (void * b : bounces_)
        if (b) pio::aligned_free(b);
    fds_.clear();
    fds_buf_.clear();
    bounces_.clear();
    bounce_sz_.clear();
}

} // namespace bmoe
