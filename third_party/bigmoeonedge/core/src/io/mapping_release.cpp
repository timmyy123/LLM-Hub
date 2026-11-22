#include "mapping_release.h"

// System headers at global scope, as in platform_io.cpp.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX // windows.h's max() macro would shadow std::max below
#endif
#include <windows.h>
#include <winternl.h>
#include <algorithm>
#include <cwctype>
#include <map>
#else
#include <sys/mman.h>
#include "platform_io.h"
#endif

namespace bmoe::pio {

MappingPlaceholders::~MappingPlaceholders() {
    release();
}

MappingPlaceholders::MappingPlaceholders(MappingPlaceholders && o) noexcept
    : reserved_(std::move(o.reserved_)), viewed_(std::move(o.viewed_)), plugs_(std::move(o.plugs_)) {
    o.reserved_.clear();
    o.viewed_.clear();
    o.plugs_.clear();
}

MappingPlaceholders & MappingPlaceholders::operator=(MappingPlaceholders && o) noexcept {
    if (this != &o) {
        release();
        reserved_ = std::move(o.reserved_);
        viewed_ = std::move(o.viewed_);
        plugs_ = std::move(o.plugs_);
        o.reserved_.clear();
        o.viewed_.clear();
        o.plugs_.clear();
    }
    return *this;
}

void MappingPlaceholders::release() {
#if defined(_WIN32)
    for (const auto & r : reserved_) {
        VirtualFree(r.first, 0, MEM_RELEASE);
    }
    // The placeholder views are the model's to unmap, and its teardown (which runs before this)
    // already has: unmapping the base again could take out whatever was mapped there since. The
    // same reasoning as the POSIX placeholders below. If the model never unmaps, a reserved view
    // with no commit behind it leaks until process exit.
#else
    // POSIX: the model's own munmap of "its" range has already removed the placeholder by the
    // time this runs (it is called after the model is freed), and unmapping the range again could
    // take out whatever was mapped there since. If the model never unmaps, an inaccessible
    // reservation leaks until process exit. Either way: forget, do not touch.
#endif
#if defined(_WIN32)
    // The plugs are deliberately NOT closed here. Each occupies a handle value the model still
    // believes is its section, and the model's own teardown closes that value; closing it here as
    // well would be a double close of whatever occupies the slot by then. If the model never
    // closes it (it always does today), one inert event per section leaks until process exit.
#endif
    reserved_.clear();
    viewed_.clear();
    plugs_.clear();
}

#if defined(_WIN32)

namespace {

std::wstring widen(const std::string & s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return {};
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), &w[0], n);
    return w;
}

std::wstring lower(std::wstring w) {
    for (wchar_t & c : w)
        c = (wchar_t) std::towlower(c);
    return w;
}

// The file's NT device path ("\Device\HarddiskVolumeN\..."), which is the form the memory manager
// reports for mapped views, so the two can be compared directly. Opened with no access rights:
// identification only, and closed before returning. Empty on failure.
std::wstring nt_path_of(const std::wstring & path) {
    HANDLE h = CreateFileW(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return {};
    wchar_t buf[32768];
    const DWORD n = GetFinalPathNameByHandleW(h, buf, (DWORD) (sizeof(buf) / sizeof(buf[0])), VOLUME_NAME_NT);
    CloseHandle(h);
    if (n == 0 || n >= sizeof(buf) / sizeof(buf[0])) return {};
    return lower(std::wstring(buf, n));
}

using GetMappedFileNameW_t = DWORD(WINAPI *)(HANDLE, LPVOID, LPWSTR, DWORD);
using NtQueryInformationProcess_t = NTSTATUS(NTAPI *)(HANDLE, PROCESSINFOCLASS, PVOID, ULONG, PULONG);
using NtQueryObject_t = NTSTATUS(NTAPI *)(HANDLE, OBJECT_INFORMATION_CLASS, PVOID, ULONG, PULONG);

// Which file a view (or any address inside one) maps, as an NT device path; empty if none.
std::wstring mapped_name(GetMappedFileNameW_t fn, void * addr) {
    wchar_t buf[32768];
    const DWORD n = fn(GetCurrentProcess(), addr, buf, (DWORD) (sizeof(buf) / sizeof(buf[0])));
    if (n == 0) return {};
    return lower(std::wstring(buf, n));
}

bool is_target(const std::vector<std::wstring> & targets, const std::wstring & name) {
    return !name.empty() && std::find(targets.begin(), targets.end(), name) != targets.end();
}

// Every view of the target files: [AllocationBase, end) per view, assembled from the regions
// VirtualQuery walks (a view can be reported as several regions with one AllocationBase).
std::vector<std::pair<void *, size_t>> find_views(GetMappedFileNameW_t fn, const std::vector<std::wstring> & targets) {
    std::map<void *, uintptr_t> ends; // AllocationBase -> highest end seen
    MEMORY_BASIC_INFORMATION mbi;
    const char * p = nullptr;
    while (VirtualQuery(p, &mbi, sizeof(mbi)) == sizeof(mbi)) {
        if (mbi.Type == MEM_MAPPED && mbi.State == MEM_COMMIT && is_target(targets, mapped_name(fn, mbi.BaseAddress))) {
            const uintptr_t end = (uintptr_t) mbi.BaseAddress + mbi.RegionSize;
            uintptr_t & e = ends[mbi.AllocationBase];
            e = std::max(e, end);
        }
        const char * next = (const char *) mbi.BaseAddress + mbi.RegionSize;
        if (next <= p) break; // wrapped: end of the address space
        p = next;
    }
    std::vector<std::pair<void *, size_t>> out;
    for (const auto & kv : ends)
        out.emplace_back(kv.first, (size_t) (kv.second - (uintptr_t) kv.first));
    return out;
}

// Hold the released range with a VIEW of an empty pagefile-backed section, so the model's own
// UnmapViewOfFile on this base at teardown is a success that releases the placeholder, exactly as
// its munmap does on POSIX. A plain reservation would also keep the range safe, but llama.cpp's
// unmap of it then fails and warns ("UnmapViewOfFile failed") on every run that released the
// mapping. SEC_RESERVE makes the section's pages reserved, not committed: the placeholder costs no
// commit charge however large the model file, and touching it faults like any reservation. The
// section handle is closed at once; the view alone keeps the section alive, so no handle is left
// for the section pass below to find. Null if the range cannot be mapped at this exact base.
void * map_placeholder(void * base, size_t size) {
    const uint64_t s = (uint64_t) size;
    HANDLE sec = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, (DWORD) (s >> 32),
                                    (DWORD) (s & 0xFFFFFFFFu), nullptr);
    if (!sec) return nullptr;
    void * v = MapViewOfFileEx(sec, FILE_MAP_READ, 0, 0, size, base);
    CloseHandle(sec);
    if (v && v != base) { // cannot happen with an explicit base, but a view elsewhere holds nothing
        UnmapViewOfFile(v);
        return nullptr;
    }
    return v;
}

// The fallback when the range cannot be held by a view: a reservation keeps it out of reach of
// any later allocation, at the price of the model's unmap failing (harmlessly) at teardown. A
// reservation is in 64 KiB units, so if the exact size collides with a neighbour the tail is given
// up rather than the whole placeholder.
void * reserve_placeholder(void * base, size_t size) {
    if (void * r = VirtualAlloc(base, size, MEM_RESERVE, PAGE_NOACCESS)) return r;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    const size_t g = si.dwAllocationGranularity;
    const size_t down = size & ~(g - 1);
    if (down && down < size) return VirtualAlloc(base, down, MEM_RESERVE, PAGE_NOACCESS);
    return nullptr;
}

struct HandleEntry {
    HANDLE HandleValue;
    ULONG_PTR HandleCount;
    ULONG_PTR PointerCount;
    ULONG GrantedAccess;
    ULONG ObjectTypeIndex;
    ULONG HandleAttributes;
    ULONG Reserved;
};
struct HandleSnapshot {
    ULONG_PTR NumberOfHandles;
    ULONG_PTR Reserved;
    HandleEntry Handles[1];
};
constexpr int kProcessHandleInformation = 51;
constexpr NTSTATUS kStatusInfoLengthMismatch = (NTSTATUS) 0xC0000004L;
constexpr DWORD kSectionMapRead = 0x0004; // SECTION_MAP_READ
constexpr ULONG kProtectFromClose = 0x2;  // HANDLE_FLAG_PROTECT_FROM_CLOSE

std::vector<char> handle_snapshot(NtQueryInformationProcess_t q) {
    std::vector<char> buf(1 << 16);
    for (int attempt = 0; attempt < 8; ++attempt) {
        ULONG need = 0;
        const NTSTATUS st =
            q(GetCurrentProcess(), (PROCESSINFOCLASS) kProcessHandleInformation, buf.data(), (ULONG) buf.size(), &need);
        if (st == kStatusInfoLengthMismatch) {
            buf.resize((size_t) need + (1 << 14));
            continue;
        }
        if (st < 0) return {};
        return buf;
    }
    return {};
}

bool is_section(NtQueryObject_t q, HANDLE h) {
    alignas(PUBLIC_OBJECT_TYPE_INFORMATION) char buf[sizeof(PUBLIC_OBJECT_TYPE_INFORMATION) + 512];
    ULONG len = 0;
    if (q(h, ObjectTypeInformation, buf, (ULONG) sizeof(buf), &len) < 0) return false;
    const auto * ti = (const PUBLIC_OBJECT_TYPE_INFORMATION *) buf;
    const size_t n = ti->TypeName.Length / sizeof(wchar_t);
    return n == 7 && std::wstring(ti->TypeName.Buffer, n) == L"Section";
}

// Re-occupy a just-closed handle value with an inert event so no later handle can inherit it and
// be closed by the model's teardown in its place. Handle values are recycled lowest-free-first, so
// the first attempt normally lands; a few more cover a concurrent allocation. Returns the plug, or
// null if the slot could not be reclaimed (then the model's late CloseHandle hits whatever lives
// there, which is why the caller reports it).
HANDLE plug_handle_slot(HANDLE value) {
    std::vector<HANDLE> misses;
    HANDLE plug = nullptr;
    for (int i = 0; i < 16 && !plug; ++i) {
        HANDLE e = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!e) break;
        if (e == value)
            plug = e;
        else
            misses.push_back(e);
    }
    for (HANDLE m : misses)
        CloseHandle(m);
    return plug;
}

} // namespace

size_t addresses_in_file_mappings(const std::vector<std::string> & paths, const std::vector<const void *> & addresses) {
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    auto mapped_fn = k32 ? (GetMappedFileNameW_t) (void *) GetProcAddress(k32, "K32GetMappedFileNameW") : nullptr;
    if (!mapped_fn) return 0; // cannot tell; the caller's other conditions still apply
    std::vector<std::wstring> targets;
    for (const std::string & p : paths) {
        std::wstring nt = nt_path_of(widen(p));
        if (!nt.empty()) targets.push_back(std::move(nt));
    }
    size_t hits = 0;
    for (const void * a : addresses) {
        if (!a) continue;
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(a, &mbi, sizeof(mbi)) != sizeof(mbi)) continue;
        if (mbi.Type != MEM_MAPPED) continue;
        if (is_target(targets, mapped_name(mapped_fn, (void *) a))) ++hits;
    }
    return hits;
}

MappingReleaseReport release_file_mappings(const std::vector<std::string> & paths, MappingPlaceholders * out) {
    MappingReleaseReport rep;
    rep.supported = true;

    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    auto mapped_fn = k32 ? (GetMappedFileNameW_t) (void *) GetProcAddress(k32, "K32GetMappedFileNameW") : nullptr;
    auto qip =
        ntdll ? (NtQueryInformationProcess_t) (void *) GetProcAddress(ntdll, "NtQueryInformationProcess") : nullptr;
    auto qobj = ntdll ? (NtQueryObject_t) (void *) GetProcAddress(ntdll, "NtQueryObject") : nullptr;
    if (!mapped_fn || !qip || !qobj) {
        rep.error = "required kernel32/ntdll entry points unavailable";
        return rep;
    }

    std::vector<std::wstring> targets;
    for (const std::string & p : paths) {
        std::wstring nt = nt_path_of(widen(p));
        if (nt.empty()) {
            if (rep.error.empty()) rep.error = "cannot identify " + p;
            continue;
        }
        targets.push_back(std::move(nt));
    }
    if (targets.empty()) return rep;

    // 1. Views. Unmapping first means the section-closing pass below cannot pull a view out from
    //    under a live mapping; each view's range is held by a placeholder afterwards.
    for (const auto & v : find_views(mapped_fn, targets)) {
        if (!UnmapViewOfFile(v.first)) {
            if (rep.error.empty()) rep.error = "UnmapViewOfFile failed";
            continue;
        }
        ++rep.views_unmapped;
        rep.bytes += v.second;
        if (void * pv = map_placeholder(v.first, v.second)) {
            if (out) out->viewed_.emplace_back(pv, v.second);
        } else if (void * r = reserve_placeholder(v.first, v.second)) {
            if (out) out->reserved_.emplace_back(r, v.second);
        } else if (rep.error.empty()) {
            rep.error = "placeholder reservation failed";
        }
    }

    // 2. Section handles. A section is matched by mapping one page of it and asking which file
    //    that page belongs to, never by guessing from handle values or types alone.
    const std::vector<char> snap = handle_snapshot(qip);
    if (snap.empty()) {
        if (rep.error.empty()) rep.error = "process handle snapshot unavailable";
        return rep;
    }
    const auto * hs = (const HandleSnapshot *) snap.data();
    for (ULONG_PTR i = 0; i < hs->NumberOfHandles; ++i) {
        const HandleEntry & e = hs->Handles[i];
        if ((e.HandleAttributes & kProtectFromClose) || !(e.GrantedAccess & kSectionMapRead)) continue;
        if (!is_section(qobj, e.HandleValue)) continue;
        void * probe = MapViewOfFile(e.HandleValue, FILE_MAP_READ, 0, 0, 1);
        if (!probe) continue;
        const bool match = is_target(targets, mapped_name(mapped_fn, probe));
        UnmapViewOfFile(probe);
        if (!match) continue;
        if (!CloseHandle(e.HandleValue)) {
            if (rep.error.empty()) rep.error = "CloseHandle on section failed";
            continue;
        }
        ++rep.sections_closed;
        if (HANDLE plug = plug_handle_slot(e.HandleValue)) {
            if (out) out->plugs_.push_back(plug);
        } else {
            ++rep.plugs_missed;
        }
    }
    return rep;
}

#else // POSIX: the file's VMAs come from /proc/self/maps; there are no handles to deal with.

// Each region is unmapped and immediately replaced by an inaccessible anonymous placeholder at the
// same address, so the model's later munmap of "its" range removes the placeholder and nothing
// else, and no allocation can land there in between. Offered for measurement: Android (f2fs) was
// measured NOT to serialise direct reads against a live mapping, so there is no throughput to
// gain there; what a release buys on a given POSIX system is a bench question.
size_t addresses_in_file_mappings(const std::vector<std::string> & paths, const std::vector<const void *> & addresses) {
    std::vector<MappedRegion> regs;
    std::vector<MappedRegion> all;
    for (const std::string & p : paths) {
        const size_t slash = p.find_last_of('/');
        const std::string base = slash == std::string::npos ? p : p.substr(slash + 1);
        regs.clear();
        if (file_mapped_regions(base.c_str(), regs)) all.insert(all.end(), regs.begin(), regs.end());
    }
    size_t hits = 0;
    for (const void * a : addresses) {
        const uintptr_t x = (uintptr_t) a;
        for (const MappedRegion & r : all)
            if (x >= r.start && x < r.end) {
                ++hits;
                break;
            }
    }
    return hits;
}

MappingReleaseReport release_file_mappings(const std::vector<std::string> & paths, MappingPlaceholders * out) {
    MappingReleaseReport rep;
    rep.supported = true;
    for (const std::string & p : paths) {
        const size_t slash = p.find_last_of('/');
        const std::string base = slash == std::string::npos ? p : p.substr(slash + 1);
        std::vector<MappedRegion> regs;
        if (!file_mapped_regions(base.c_str(), regs)) {
            if (rep.error.empty()) rep.error = "cannot read /proc/self/maps";
            continue;
        }
        for (const MappedRegion & r : regs) {
            void * addr = (void *) r.start;
            const size_t size = (size_t) (r.end - r.start);
            if (munmap(addr, size) != 0) {
                if (rep.error.empty()) rep.error = "munmap failed";
                continue;
            }
            ++rep.views_unmapped;
            rep.bytes += size;
            void * ph = mmap(addr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
            if (ph == MAP_FAILED) {
                if (rep.error.empty()) rep.error = "placeholder mapping failed";
            } else if (out) {
                out->reserved_.emplace_back(ph, size);
            }
        }
    }
    return rep;
}

#endif

} // namespace bmoe::pio
