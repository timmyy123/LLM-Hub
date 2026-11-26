// Classifier-free guidance + sampling, one token from logits[0,count) → tokbuf[q].
//
// This is sample_range_f32 with the step the port was missing. Upstream runs the depthformer over a
// BATCH of conditionings — the real one plus one per masked input — and combines their logits before
// it samples (magenta_rt/mlx/depthformer.py:247-257):
//
//     cfg_logits = pos + Σ_i scale_i · (pos − neg_i)
//
// with scales {musiccoca: 3.0, notes: 1.0} (mlx/system.py). At those values the conditional
// direction is amplified ~5x before a temperature-1.3 draw. This port sampled from `pos` alone,
// i.e. scale 0, which is a far flatter distribution — measured on the reference oracle, paired over
// 12 prompts, that costs 0.071 of prompt-discrimination margin with the reference ahead in 11 of 12.
//
// ORDER MATTERS and is the reference's:  soft cap (per stream) → CFG → top-k → Gumbel → argmax.
//   * The cap is applied to EACH stream before combining. Upstream caps inside the decoder
//     (mlx/depthformer.py:524-527), which runs on the whole batch, so every stream is capped and the
//     combine sees capped values. Capping after the combine would cap a 5x-larger number and
//     flatten exactly the separation CFG just created.
//   * Top-k runs on the COMBINED logits. Running it on `pos` first would let CFG promote a token
//     that masking had already discarded — a different distribution, not a reordering.
//
// s1 = 0 and s2 = 0 reduce this to sample_range_f32 exactly (the neg terms vanish), so the
// single-stream path stays bit-identical and l1/l2 may alias l0 when unused.
#define SAMPLE_WG 64

// Counter-based RNG, identical to sample_range_f32 so a seed reproduces the same draw on either
// path — the two kernels must not disagree about what "seed 7" means.
inline uint src_hash(uint a, uint b, uint c) {
  uint x = a * 0x9E3779B9u ^ b * 0x85EBCA6Bu ^ c * 0xC2B2AE35u;
  x ^= x >> 16; x *= 0x7FEB352Du;
  x ^= x >> 15; x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

// Capped-and-combined logit at index i. Reads all three streams unconditionally: a branch here
// would diverge the wavefront on every element, and when a stream is unused its pointer aliases l0
// and its scale is 0, so the term is exactly zero rather than merely small.
inline float src_cfg(__global const float* l0, __global const float* l1, __global const float* l2,
                     const int i, const float s1, const float s2, const float cap) {
  float a0 = l0[i], a1 = l1[i], a2 = l2[i];
  if (cap > 0.0f) {
    a0 = tanh(a0 / cap) * cap;
    a1 = tanh(a1 / cap) * cap;
    a2 = tanh(a2 / cap) * cap;
  }
  return a0 + s1 * (a0 - a1) + s2 * (a0 - a2);
}

__kernel __attribute__((reqd_work_group_size(SAMPLE_WG, 1, 1)))
void sample_range_cfg_f32(__global const float* l0,      // positive conditioning
                          __global const float* l1,      // negative 1 (aliases l0 when s1 == 0)
                          __global const float* l2,      // negative 2 (aliases l0 when s2 == 0)
                          __global int* tokbuf,
                          const int base, const int count, const int q,
                          const float s1, const float s2,
                          const float temperature, const int top_k,
                          const float soft_cap, const uint seed,
                          // The frame index arrives in a DEVICE buffer, not as a host-set uint. A
                          // recorded frame holds no copy of its args, so a host-baked value here
                          // would freeze at the captured frame and every replay would draw the
                          // same Gumbel noise forever — see nnopt_pos_buffer, same mechanism `pos`
                          // already uses. Value is identical (pos0+f), so sampling is unchanged.
                          __global const int* frame_buf){
  __local float lv[SAMPLE_WG];
  __local int   li[SAMPLE_WG];
  __local float kth;
  const int t = get_local_id(0);
  const uint frame = (uint)frame_buf[0];

  // ── 1. top-k threshold on the COMBINED logits: k passes of "max below the last max" ──
  if (top_k > 0 && top_k < count) {
    float bound = INFINITY;
    for (int pass = 0; pass < top_k; ++pass) {
      float bv = -INFINITY;
      for (int i = t; i < count; i += SAMPLE_WG) {
        const float v = src_cfg(l0, l1, l2, i, s1, s2, soft_cap);
        if (v < bound && v > bv) bv = v;
      }
      lv[t] = bv;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int s = SAMPLE_WG >> 1; s > 0; s >>= 1) {
        if (t < s && lv[t + s] > lv[t]) lv[t] = lv[t + s];
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      bound = lv[0];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) kth = bound;
  } else {
    if (t == 0) kth = -INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float thresh = kth;

  // ── 2. temperature, Gumbel noise, argmax ──
  float bv = -INFINITY; int bi = 0;
  for (int i = t; i < count; i += SAMPLE_WG) {
    float v = src_cfg(l0, l1, l2, i, s1, s2, soft_cap);
    if (v < thresh) continue;                    // outside top-k: impossible
    if (temperature > 0.0f) {
      const uint  h = src_hash((uint)i, (uint)q ^ seed, frame);
      const float u = ((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
      v = v / temperature + (-log(-log(u)));
    }
    if (v > bv) { bv = v; bi = i; }
  }
  lv[t] = bv; li[t] = bi;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = SAMPLE_WG >> 1; s > 0; s >>= 1) {
    if (t < s) {
      const float ov = lv[t + s];
      // Ties keep the lower index, matching argmax_range_f32 and sample_range_f32.
      if (ov > lv[t] || (ov == lv[t] && li[t + s] < li[t])) { lv[t] = ov; li[t] = li[t + s]; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) tokbuf[q] = base + li[0];
}
