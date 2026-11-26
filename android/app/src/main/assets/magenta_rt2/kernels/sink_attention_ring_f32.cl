// Cross-attention over the CONDITIONING RING — [sink, source_0 .. source_{n_valid-1}].
//
// This is the faithful form of what sink_attention_wg_f32.cl approximates with its `log_n` trick.
// Upstream (use_streaming_cross_attention, max_past_horizon=41 for mrt2_small) keeps a KV ring of
// the last 41 conditioning frames plus the current one, so a step attends over up to 42 sources.
//
// The ln(N) shortcut is EXACT while the conditioning is constant, because all N keys are then
// bit-identical. It stops being exact the moment the conditioning MOVES — a prompt blend being
// dragged — and that is precisely when it matters: upstream cross-fades a conditioning change over
// 42 frames (1.68 s) because the ring still holds the old vectors, while a single-source
// approximation switches instantly. This kernel holds the real thing.
//
// Ring layout is [slot][h][d]; slots are written round-robin at pos % S. Attention is
// permutation-invariant over keys, so slot ORDER carries no meaning — only which slots are valid.
// Before the ring has filled, the valid slots are exactly 0..n_valid-1 (they are written in
// order from 0), which is why no separate validity mask is needed.
//
// One workgroup per head. Phase 2 gives each key its own work-item, so the n_valid dot products
// run in parallel with no reduction — the D-loop stays inside one thread. That keeps the whole
// kernel to three barriers regardless of ring depth.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 64
#define MAXD 256           // units_per_head ceiling (temporal cross-attn is 128)
#define MAXS (WG - 1)      // one work-item per key in phase 2, so n_valid must fit the group

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void sink_attention_ring_f32(
    __global const float* q,             // [H*D]   query, already projected
    __global const float* ring_k,        // [S*H*D] projected keys,  slot-major
    __global const float* ring_v,        // [S*H*D] projected values, slot-major
    __global const half*  sink_k,        // [H*D]
    __global const half*  sink_v,        // [H*D]
    __global const half*  per_dim_scale, // [D]
    __global float*       out,           // [H*D]
    const int H,
    const int D,
    const float inv_sqrt_d,
    const int n_valid,                   // 1 .. S
    const int S) {                       // ring depth (stride between slots)
  const int h   = get_group_id(0);
  const int lid = get_local_id(0);
  const int HD  = H * D;
  const size_t qb = (size_t)h * D;

  __local float qs[MAXD];        // per-dim-scaled query, shared by every key
  __local float lg[MAXS + 1];    // [0] = sink logit, [1 + s] = slot s

  const int nv = (n_valid < 1) ? 1 : (n_valid > MAXS ? MAXS : n_valid);

  // r_softplus_0 = 1/ln2 normalises so per_dim_scale=0 -> scale=query_scale (attention.py:38).
  const float r_softplus_0 = 1.442695041f;
  for (int d = lid; d < D; d += WG) {
    const float sv = r_softplus_0 * inv_sqrt_d * log1p(exp(vload_half(d, per_dim_scale)));
    qs[d] = q[qb + d] * sv;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // The sink logit uses the UNSCALED query (JAX behaviour, mirrored in the two-key kernels).
  if (lid == 0) {
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += q[qb + d] * vload_half(qb + d, sink_k);
    lg[0] = s;
  } else if (lid <= nv) {
    const size_t kb = (size_t)(lid - 1) * HD + qb;
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += qs[d] * ring_k[kb + d];
    lg[lid] = s;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // nv+1 <= 64 values — cheaper to softmax them serially in one lane than to run a tree reduction.
  if (lid == 0) {
    float mx = lg[0];
    for (int i = 1; i <= nv; ++i) mx = lg[i] > mx ? lg[i] : mx;
    float z = 0.0f;
    for (int i = 0; i <= nv; ++i) { lg[i] = exp(lg[i] - mx); z += lg[i]; }
    const float inv = 1.0f / z;
    for (int i = 0; i <= nv; ++i) lg[i] *= inv;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int d = lid; d < D; d += WG) {
    float acc = lg[0] * vload_half(qb + d, sink_v);
    for (int s = 0; s < nv; ++s) acc += lg[1 + s] * ring_v[(size_t)s * HD + qb + d];
    out[qb + d] = acc;
  }
}
