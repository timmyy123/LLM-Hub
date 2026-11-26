// Attention softmax for MusicCoCa's text tower: padding mask + logit capping.
//
// MusicCoCa caps attention logits before the softmax — `cap * tanh(logits / cap)` — which is why
// the exported graph has a TANH between the QK^T matmul and the softmax. Skipping it changes the
// distribution on any prompt with strong matches, so it is part of the model, not a nicety.
//
// logits [H, T, T] fp32 in place. `valid[j] != 0` marks a real (non-padding) key position; masked
// positions are driven to -inf BEFORE the max, so they cannot contribute to the normalisation.
//
// One work-group per (head, query row): a row of T keys is reduced in local memory.
__kernel void softmax_mask_cap_f32(__global float* logits,      // [H, T, T]
                                   __global const int* valid,   // [T]
                                   const int T,
                                   const float cap,
                                   const float scale) {
  const int row = get_group_id(0);            // head*T + query index
  const int lid = get_local_id(0);
  const int lsz = get_local_size(0);
  __global float* r = logits + (size_t)row * T;
  __local float red[256];

  // scale → cap → mask
  for (int j = lid; j < T; j += lsz) {
    float x = r[j] * scale;
    if (cap > 0.0f) x = cap * tanh(x / cap);
    r[j] = (valid[j] != 0) ? x : -INFINITY;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // max
  float m = -INFINITY;
  for (int j = lid; j < T; j += lsz) m = fmax(m, r[j]);
  red[lid] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsz / 2; s > 0; s >>= 1) {
    if (lid < s) red[lid] = fmax(red[lid], red[lid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  m = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // exp + sum
  float s0 = 0.0f;
  for (int j = lid; j < T; j += lsz) {
    const float e = (r[j] == -INFINITY) ? 0.0f : exp(r[j] - m);
    r[j] = e;
    s0 += e;
  }
  red[lid] = s0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsz / 2; s > 0; s >>= 1) {
    if (lid < s) red[lid] += red[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = 1.0f / fmax(red[0], 1e-20f);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int j = lid; j < T; j += lsz) r[j] *= inv;
}
