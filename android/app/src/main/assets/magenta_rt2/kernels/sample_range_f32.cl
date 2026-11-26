// Sample one token from logits[0,count) → writes (base + index) into tokbuf[q].
//
// Replaces plain argmax. Greedy decoding of an RVQ audio LM is a well-known degeneracy: with no
// randomness the model can reach a fixed point and never leave it, which is heard as a short figure
// repeating forever — "like a CD stuck". The reference samples instead
// (magenta_rt/mlx/system.py: temperature = 1.3, top_k = 40).
//
// Three things happen here, in the order the reference does them:
//
//   1. SOFT CAP.  x = tanh(x / cap) * cap, with cap = soft_cap_logits = 30.0 from the model config
//      (magenta_rt/jax/model.py, MultivariateDecoder.Config). This port skipped it on the grounds
//      that a monotonic transform cannot change an argmax — true, and false the moment temperature
//      exists, because the cap changes the SPACING between logits and therefore every probability.
//      Skipping it here would sample from the wrong distribution.
//
//   2. TOP-K.  Find the k-th largest value and treat everything below it as impossible. Done with a
//      k-pass selection over the workgroup rather than a sort: k is 40 and count is 1024, so this is
//      cheaper than sorting and needs no scratch.
//
//   3. GUMBEL-MAX.  argmax(logit/T + g), g = -log(-log(u)). This draws exactly from the softmax
//      distribution at temperature T without ever forming it — no exp, no normalisation, no
//      cumulative sum, and it reuses the reduction the argmax already had.
//
// temperature <= 0 restores exact greedy behaviour, so the old path is still reachable.
#define SAMPLE_WG 64

// Counter-based RNG (Philox-style mixing). A stateless hash of (seed, q, frame) beats carrying a
// generator: every draw is reproducible from its coordinates, so a render can be replayed exactly
// by passing the same seed, and no state has to survive a chunk boundary or a device reset.
inline uint sr_hash(uint a, uint b, uint c) {
  uint x = a * 0x9E3779B9u ^ b * 0x85EBCA6Bu ^ c * 0xC2B2AE35u;
  x ^= x >> 16; x *= 0x7FEB352Du;
  x ^= x >> 15; x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

__kernel __attribute__((reqd_work_group_size(SAMPLE_WG, 1, 1)))
void sample_range_f32(__global const float* logits, __global int* tokbuf,
                      const int base, const int count, const int q,
                      const float temperature, const int top_k,
                      const float soft_cap, const uint seed,
                      // Device buffer, not a host-set uint: a recording holds no copy of its args,
                      // so a host-baked frame would freeze at the captured frame and every replay
                      // would redraw the same noise. Same mechanism `pos` already uses.
                      __global const int* frame_buf){
  __local float lv[SAMPLE_WG];
  __local int   li[SAMPLE_WG];
  __local float kth;
  const int t = get_local_id(0);
  const uint frame = (uint)frame_buf[0];

  // ── 1. top-k threshold: the k-th largest logit, by k passes of "max below the last max" ──
  // Skipped when k covers the whole slice, which is the common case for k >= count.
  if (top_k > 0 && top_k < count) {
    float bound = INFINITY;
    for (int pass = 0; pass < top_k; ++pass) {
      float bv = -INFINITY;
      for (int i = t; i < count; i += SAMPLE_WG) {
        float v = logits[i];
        if (soft_cap > 0.0f) v = tanh(v / soft_cap) * soft_cap;
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

  // ── 2 + 3. soft cap, temperature, Gumbel noise, argmax ──
  float bv = -INFINITY; int bi = 0;
  for (int i = t; i < count; i += SAMPLE_WG) {
    float v = logits[i];
    if (soft_cap > 0.0f) v = tanh(v / soft_cap) * soft_cap;
    if (v < thresh) continue;                    // outside top-k: impossible
    if (temperature > 0.0f) {
      // u in (0,1): 0 would make log(0) = -inf and 1 would make -log(1) = 0 collapse the draw.
      const uint  h = sr_hash((uint)i, (uint)q ^ seed, frame);
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
      // Ties keep the lower index, matching argmax_range_f32 so temperature=0 is bit-identical.
      if (ov > lv[t] || (ov == lv[t] && li[t + s] < li[t])) { lv[t] = ov; li[t] = li[t + s]; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) tokbuf[q] = base + li[0];
}
