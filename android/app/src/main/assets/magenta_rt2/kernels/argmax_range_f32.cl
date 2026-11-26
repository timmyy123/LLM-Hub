// argmax over logits[0,count) → writes (base+argmax) into tokbuf[q]. (logits is the codebook slice.)
//
// Was a single work item walking all `count` elements: 83 us/call measured on an Adreno 840 for
// count=1024, 12 calls per AR frame. One lane of one compute unit cannot hide any of that latency
// (the Adreno guide notes a workgroup never spreads across compute units), so it was ~6% of the
// frame's kernel time to pick 12 maxima. One 64-wide workgroup with a local reduction instead.
//
// Ties: the serial version kept the FIRST maximum (strict >). The reduction preserves that by
// keeping the lower index whenever two values compare equal — otherwise a tie could pick a
// different token than the sequential path and silently change the audio.
#define ARGMAX_WG 64

__kernel __attribute__((reqd_work_group_size(ARGMAX_WG, 1, 1)))
void argmax_range_f32(__global const float* logits, __global int* tokbuf,
                      const int base, const int count, const int q){
  __local float lv[ARGMAX_WG];
  __local int   li[ARGMAX_WG];
  const int t = get_local_id(0);

  float bv = -INFINITY; int bi = 0;
  for (int i = t; i < count; i += ARGMAX_WG) {
    const float v = logits[i];
    if (v > bv) { bv = v; bi = i; }
  }
  lv[t] = bv; li[t] = bi;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = ARGMAX_WG >> 1; s > 0; s >>= 1) {
    if (t < s) {
      const float ov = lv[t + s];
      // `>` alone would let the higher-index half win a tie; compare indices to keep the first.
      if (ov > lv[t] || (ov == lv[t] && li[t + s] < li[t])) { lv[t] = ov; li[t] = li[t + s]; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (t == 0) tokbuf[q] = base + li[0];
}
