// Cached self-attention, ONE WORKGROUP PER HEAD (vs one work ITEM per head).
//
// Same math as self_attn_cached_f32.cl. Two things were wrong with the original:
//
//  1. It was dispatched with global size H (12) and no local size — twelve fibers for the whole
//     GPU. Adreno pins a workgroup to one compute unit and cannot split it (80-NB295-11 §3.2.5),
//     so twelve work items is twelve lanes: ~1/64th of one CU, and eleven CUs idle. 826 us/call.
//  2. The per-dimension scale is `r * inv_sqrt_d * softplus(pds[d])` — a function of d ALONE — but
//     it was recomputed inside the position loop, so the exp/log1p pair ran once per (position,
//     dim) pair instead of once per dim. At ~41 cached positions that is ~41x the transcendental
//     work, and it is the same value every time.
//
// Now: WG threads per head. Scale is computed once into local memory, scores are computed one
// position per thread, and the output is written one dim per thread.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 64
#define MAXD 256          // per-head dim; guarded by the caller
#define MAXPOS 160        // sink + cached positions; guarded by the caller

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void self_attn_cached_wg_f32(
    // kcache/vcache are WRITABLE because append_kv folds kv_append_f32 into this kernel.
    __global const float* qkv, __global float* kcache, __global float* vcache,
    __global const half* sink_k, __global const half* sink_v, __global const half* pds,
    __global float* out, const int H, const int D,
    // pos comes from a BUFFER so one recorded dispatch can serve every frame — see kv_append_f32.cl.
    // max_past_in < 0 means "unbounded" (the depth body), which resolves to pos+1 here rather than
    // on the host, since the host value would be baked into a recording.
    __global const int* posb, const int has_sink,
    const float inv_sqrt_d, const int max_past_in,
    // append_kv=1 folds kv_append_f32 into this kernel, removing one dispatch per attention site
    // (36/frame, 2.2 us of GPU work each against ~55 us of HOST cost to launch — a 25:1 overhead
    // ratio). It is safe at WORKGROUP scope and needs no global barrier: the cache is laid out
    // [pos][head][dim], workgroup h writes only head h's D-element slice at `pos`, and the
    // attention below reads only head h's slice. No workgroup ever reads another's write, so the
    // local barrier that already exists before the score loop is sufficient ordering.
    const int append_kv){
  const int pos = posb[0];
  const int max_past = (max_past_in < 0) ? (pos + 1) : max_past_in;
  const int h   = get_group_id(0);
  const int lid = get_local_id(0);
  if (h >= H) return;
  const int HD = H * D;
  const size_t qb = (size_t)h * D;
  const float r = 1.442695041f;

  int lo = pos - max_past; if (lo < 0) lo = 0;
  const int ns = has_sink ? 1 : 0;
  const int n  = ns + (pos - lo + 1);

  // ── folded kv_append ────────────────────────────────────────────────────────────────────────
  // Identical arithmetic to kv_append_f32.cl: k starts one HD block into qkv, v two blocks in.
  // Strided over the workgroup because D (<=MAXD) may exceed WG.
  if (append_kv) {
    const size_t dstb = (size_t)pos * (size_t)HD + qb;
    for (int d = lid; d < D; d += WG) {
      kcache[dstb + d] = qkv[HD + qb + d];
      vcache[dstb + d] = qkv[2 * HD + qb + d];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  __local float sc_l[MAXD];        // r * inv_sqrt_d * softplus(pds[d]) — computed ONCE
  __local float scores[MAXPOS];
  __local float red;

  for (int d = lid; d < D; d += WG) sc_l[d] = r * inv_sqrt_d * log1p(exp(vload_half(d, pds)));
  barrier(CLK_LOCAL_MEM_FENCE);

  // One position per thread. The sink score uses the UNSCALED query, matching the reference.
  if (has_sink && lid == 0) {
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += qkv[qb + d] * vload_half(qb + d, sink_k);
    scores[0] = s;
  }
  for (int i = lid + ns; i < n; i += WG) {
    const int p = lo + (i - ns);
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += (qkv[qb + d] * sc_l[d]) * kcache[(size_t)p * HD + qb + d];
    scores[i] = s;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // n <= MAXPOS, so the softmax normalisation is cheap enough to do on one thread; splitting it
  // would cost two more barriers for ~160 adds. Summation order matches the serial kernel, which
  // keeps the result bit-identical — this feeds an argmax, so a last-ulp change can flip a token.
  if (lid == 0) {
    float maxs = -1e30f;
    for (int i = 0; i < n; ++i) maxs = fmax(maxs, scores[i]);
    float Z = 0.0f;
    for (int i = 0; i < n; ++i) { scores[i] = exp(scores[i] - maxs); Z += scores[i]; }
    red = Z;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float Z = red;

  for (int d = lid; d < D; d += WG) {
    float acc = 0.0f;
    if (has_sink) acc += scores[0] * vload_half(qb + d, sink_v);
    for (int p = lo; p <= pos; ++p) acc += scores[ns + (p - lo)] * vcache[(size_t)p * HD + qb + d];
    out[qb + d] = acc / Z;
  }
}
