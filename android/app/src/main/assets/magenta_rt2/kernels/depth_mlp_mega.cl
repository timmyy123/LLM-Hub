// ── STEP 2: the whole depth MLP in ONE dispatch ────────────────────────────────────────────────
//
// Today an MLP block costs two enqueues: dense1 (with the pre-norm folded in and a GELU epilogue)
// then dense2. The AR is CPU-dispatch-bound — 434 dispatches/frame at ~70 us of HOST time each,
// against 18 ms of actual GPU work — so the pair is ~140 us of CPU to do ~40 us of GPU work.
// Here they become one dispatch separated by a software global barrier.
//
// This is the smallest scope that exercises everything the depth-loop megakernel will need:
//   * a global barrier between two dependent GEMVs                (proven: BARPROBE)
//   * an intermediate passed between workgroups through global    (proven: BARVIS, 0 mismatches)
//   * register pressure in a kernel holding a full GEMV inner loop  <- the thing this is FOR
// If accumulators spill here, they will spill worse in the full loop, and this finds out cheaply.
// The codebase has measured that cliff before: an indexed private array went to scratch and cost
// 18x (407 -> 7531 us/call).
//
// PARALLELISATION. The first version gave each LANE its own output row and let it walk a whole dot
// product — no LDS tree, no local barrier in the hot loop. It was CORRECT and 4.6x SLOWER
// (ar_sec 1.7 -> 7.98, us_disp 63 -> 375). The reason is memory, not math: adjacent lanes then read
// weight addresses in_dim*2 bytes apart, so every vload_half4 in a wave touches a different cache
// line and a memory-bound GEMV falls apart.
//
// So this mirrors what linear_bias_fused_f32.cl already proved on this GPU: a workgroup owns a TILE
// of NOUT outputs, its lanes split the REDUCTION over in_dim so consecutive lanes read consecutive
// addresses, and one LDS tree reduces all NOUT accumulators together. The only new thing is that
// the workgroup loops over several tiles (a global barrier caps us at ~48 workgroups, while dense1
// wants 384 tiles), and that the two GEMVs are separated by that barrier instead of by an enqueue.
//
// Summation order still differs from the split path (tiles are strided across fewer workgroups), so
// bit-exactness is NOT expected — the gate is token/audio equivalence, not a crc match.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef SPIN_LIMIT
#define SPIN_LIMIT 2000000
#endif

// Same barrier as kernels/barrier_probe.cl — see that file for why every access is an atomic and
// why the spin is bounded. Duplicated rather than included: these are built as standalone programs.
inline int nnopt_gbarrier(__global volatile int* bar, __local int* ok,
                          const int G, const int lid, int* sense) {
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (lid == 0) {
        const int want = 1 - *sense;
        if (atomic_inc((__global int*)&bar[0]) == G - 1) {
            atomic_xchg((__global int*)&bar[0], 0);
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            atomic_xchg((__global int*)&bar[1], want);
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

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void depth_mlp_mega(
    __global const float*  x,          // [dim]  block input (un-normalised)
    __global const half*   rms_scale,  // [dim]  pre-norm weight
    __global const half*   W1,         // [hid][dim]
    __global const half*   b1,         // [hid]
    __global const half*   W2,         // [dim][hid]
    __global const half*   b2,         // [dim]
    __global float*        scratch,    // [hid]  dense1 output, crosses the barrier
    __global float*        out,        // [dim]
    __global volatile int* bar,        // [0]=count [1]=sense [2]=abort
    const int dim, const int hid, const float eps, const int G)
{
#define WG 64
// NOUT is the tile: outputs a workgroup computes at once, and therefore live accumulators per lane.
// It is the direct lever on register pressure. At NOUT=8 this kernel reported private=144B — it is
// SPILLING, and spill is the 18x cliff this codebase has hit before. Smaller tiles mean fewer live
// floats and better occupancy (which is the other half of why the megakernel only broke even), at
// the cost of re-reading x more times. The host builds several and logs each one's footprint.
#ifndef NOUT
#define NOUT 4
#endif
#if   NOUT == 4
#define REP(M) M(0) M(1) M(2) M(3)
#elif NOUT == 8
#define REP(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#elif NOUT == 16
#define REP(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) \
               M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#else
#error "NOUT must be 4, 8 or 16"
#endif
    __local float ls[WG * (NOUT + 1)];      // NOUT accumulators + one lane-set for sum(x^2)
    __local int   ok;
    const int lid = get_local_id(0), wg = get_group_id(0);
    const int dim4 = dim >> 2, hid4 = hid >> 2;

    // ── phase 1: dense1 (pre-norm folded) + GELU -> scratch ────────────────────────────────────
    for (int n0 = wg * NOUT; n0 < hid; n0 += G * NOUT) {
#define DECL(u) float a##u = 0.0f;
        REP(DECL)
        float ss = 0.0f;
        for (int j = lid; j < dim4; j += WG) {
            float4 xv = vload4(j, x);
            ss += dot(xv, xv);                       // sum of squares uses RAW x
            xv *= vload_half4(j, rms_scale);
#define MAC(u) a##u += dot(xv, vload_half4(j, W1 + (size_t)(n0 + u) * dim));
            REP(MAC)
        }
#define STORE(u) ls[lid + (u) * WG] = a##u;
        REP(STORE)
        ls[lid + NOUT * WG] = ss;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int r = WG >> 1; r > 0; r >>= 1) {
            if (lid < r) {
#define RED(u) ls[lid + (u) * WG] += ls[lid + r + (u) * WG];
                REP(RED)
                ls[lid + NOUT * WG] += ls[lid + r + NOUT * WG];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float inv = rsqrt(ls[NOUT * WG] / (float)dim + eps);
        if (lid < NOUT && n0 + lid < hid) {
            float v = ls[lid * WG] * inv + vload_half(n0 + lid, b1);
            scratch[n0 + lid] = 0.5f * v * (1.0f + erf(v * 0.70710678118654752440f));
        }
        barrier(CLK_LOCAL_MEM_FENCE);            // ls is reused by the next tile
    }

    // ── the barrier: every workgroup's dense1 slice must be visible before dense2 reads it ─────
    int sense = atomic_add((__global int*)&bar[1], 0);   // adopt the published sense, so the host
                                                         // never has to re-zero between dispatches
    if (!nnopt_gbarrier(bar, &ok, G, lid, &sense)) return;

    // ── phase 2: dense2 -> out ─────────────────────────────────────────────────────────────────
    for (int n0 = wg * NOUT; n0 < dim; n0 += G * NOUT) {
#define DECL2(u) float c##u = 0.0f;
        REP(DECL2)
        for (int j = lid; j < hid4; j += WG) {
            const float4 sv = vload4(j, scratch);
#define MAC2(u) c##u += dot(sv, vload_half4(j, W2 + (size_t)(n0 + u) * hid));
            REP(MAC2)
        }
#define STORE2(u) ls[lid + (u) * WG] = c##u;
        REP(STORE2)
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int r = WG >> 1; r > 0; r >>= 1) {
            if (lid < r) { REP(RED) }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid < NOUT && n0 + lid < dim)
            out[n0 + lid] = ls[lid * WG] + vload_half(n0 + lid, b2);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
