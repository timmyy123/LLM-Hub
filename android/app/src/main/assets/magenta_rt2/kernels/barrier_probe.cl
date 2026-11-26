// ── software GLOBAL barrier, and the probe that proves it is safe on this device ───────────────
//
// A megakernel (one dispatch for a whole depth-body RVQ loop instead of ~22 of them) needs all its
// workgroups to sync mid-kernel. OpenCL has no global barrier: barrier() only spans a workgroup.
// The standard construction is a sense-reversing counter in global memory — which is correct ONLY
// if every workgroup is simultaneously RESIDENT. If workgroup N is queued behind workgroup 0
// waiting for a CU, workgroup 0 spins for it forever and the GPU hangs.
//
// That is not a hypothetical on this hardware. The XGEMM tuner notes record a workgroup shape that
// "took the device down on first execution", and the recordable-queue work reset the GPU on both
// the 620 and the 840. So this file is built around ONE non-negotiable property:
//
//     NO SPIN IS EVER UNBOUNDED.
//
// On timeout a workgroup raises bar[2] (abort); every other workgroup sees it and leaves. The worst
// case is "the probe reports failure" — never a hung queue, never a device reset. The host then
// permanently declines to use the barrier and the ordinary split-kernel path runs.
//
// SPIN_LIMIT is sized in ITERATIONS but the constraint is TIME: it must stay far under the Android
// GPU watchdog (~2-8 s). A few million spins is single-digit milliseconds — three orders of margin.
#ifndef SPIN_LIMIT
#define SPIN_LIMIT 2000000
#endif

// bar[0] = arrival count, bar[1] = sense, bar[2] = abort. The host zeroes all three per dispatch.
// Returns 1 if the barrier completed, 0 if it aborted — callers MUST check and bail out.
// EVERY read and write of the shared state is an ATOMIC, deliberately.
//
// The first version released with a plain store to bar[1] and spun on a plain load. It passed at
// G<=4 and failed instantly at G>=6 on the 840 (measured 2026-09-12). That is the signature of a
// stale per-CU L1: `volatile` in OpenCL C orders accesses, it does NOT promise the load leaves the
// core's own cache. Workgroups placed on nearby CUs happened to observe the release; distant ones
// spun on a cached copy until the bound tripped. atomic_* goes to the coherent point, so the
// release is actually visible. Reading with atomic_add(p,0) is the idiomatic OpenCL 1.2 way to get
// a coherent load when there is no atomic_load.
inline int nnopt_gbarrier(__global volatile int* bar, __local int* ok,
                          const int G, const int lid, int* sense) {
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (lid == 0) {
        const int want = 1 - *sense;
        if (atomic_inc((__global int*)&bar[0]) == G - 1) {
            atomic_xchg((__global int*)&bar[0], 0);      // re-arm for the next round
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_xchg((__global int*)&bar[1], want);   // release everyone, coherently
            *ok = 1;
        } else {
            int spins = 0;
            for (;;) {
                if (atomic_add((__global int*)&bar[1], 0) == want) { *ok = 1; break; }
                if (atomic_add((__global int*)&bar[2], 0) != 0)    { *ok = 0; break; }
                if (++spins > SPIN_LIMIT) { atomic_xchg((__global int*)&bar[2], 1); *ok = 0; break; }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    *sense = 1 - *sense;
    return *ok;
}

// LDS_BALLAST makes the probe consume the same local memory a real megakernel would, so the
// residency bound is measured at REALISTIC occupancy. Without it the probe is a trivial kernel
// (local=4B, private=48B) that packs ~4 workgroups per CU, and reports a bound no real kernel can
// hold. The device has 32 KB of local memory per CU, so ballast B implies floor(32768/B)
// workgroups per CU and the safe G should fall as B rises. That relationship is the thing being
// measured — if it does not appear, the barrier is not residency-limited at all.
#ifndef LDS_BALLAST
#define LDS_BALLAST 0
#endif

// The probe. Runs `rounds` barriers across G workgroups and reports how many completed.
// out[0] = rounds completed (== rounds on success), out[1] = abort flag observed.
__kernel void barrier_probe(__global volatile int* bar,
                            __global int* out,
                            const int G, const int rounds) {
    __local int ok;
#if LDS_BALLAST > 0
    __local char ballast[LDS_BALLAST];
#endif
    const int lid = get_local_id(0);
    int sense = 0, done = 0;
#if LDS_BALLAST > 0
    // Fill it, then FOLD IT INTO THE OUTPUT below. A write-only array with a no-op reader is dead
    // code and the compiler removes it (measured: local stayed 4B at every ballast level), which
    // silently turns an occupancy experiment into four copies of the trivial kernel.
    for (int i = lid; i < LDS_BALLAST; i += get_local_size(0)) ballast[i] = (char)(i * 31 + rounds);
    barrier(CLK_LOCAL_MEM_FENCE);
    int bsum = 0;
    for (int i = lid; i < LDS_BALLAST; i += get_local_size(0)) bsum += ballast[i];
#endif
    for (int r = 0; r < rounds; ++r) {
        if (!nnopt_gbarrier(bar, &ok, G, lid, &sense)) break;
        done = r + 1;
    }
        if (lid == 0 && get_group_id(0) == 0) { out[0] = done; out[1] = atomic_add((__global int*)&bar[2], 0); }
#if LDS_BALLAST > 0
    // A real data dependency from the local array to a global store. The host ignores out[2];
    // its only job is to make the allocation unremovable.
    atomic_add((__global int*)&out[2], bsum & 1);
#endif
}

// ── cross-workgroup DATA visibility ────────────────────────────────────────────────────────────
// The barrier proves the FLAG becomes visible. A megakernel needs much more than that: workgroup A
// writes part of an intermediate (say dense1's output) and workgroup B reads it after the barrier.
// OpenCL 1.2 only promises mem_fence orders accesses WITHIN a work-item; nothing in the standard
// says A's ordinary global store has left A's L1 by the time B loads it. Making the flag atomic
// fixed the flag — it says nothing about the payload.
//
// So: every workgroup fills its own slice with a known pattern, barriers, then verifies its
// NEIGHBOUR's slice. out[1] is the number of mismatches; anything but 0 means a megakernel cannot
// pass data across the barrier with plain loads and stores, and the whole approach needs rethinking
// (or every access becomes atomic, which would cost more than the dispatches it saves).
__kernel void visibility_probe(__global volatile int* bar, __global int* data,
                               __global int* out, const int G, const int N) {
    __local int ok;
    const int lid = get_local_id(0), wg = get_group_id(0), L = get_local_size(0);
    int sense = 0;

    for (int i = lid; i < N; i += L) data[wg*N + i] = wg*1000003 + i*7 + 1;

    if (!nnopt_gbarrier(bar, &ok, G, lid, &sense)) {
        if (lid == 0 && wg == 0) out[0] = -1;      // barrier aborted; verdict is "unknown"
        return;
    }

    const int nb = (wg + 1) % G;                   // read someone ELSE's slice
    int bad = 0;
    for (int i = lid; i < N; i += L)
        if (data[nb*N + i] != nb*1000003 + i*7 + 1) bad++;
    if (bad) atomic_add((__global int*)&out[1], bad);
    if (lid == 0 && wg == 0) out[0] = 1;
}
