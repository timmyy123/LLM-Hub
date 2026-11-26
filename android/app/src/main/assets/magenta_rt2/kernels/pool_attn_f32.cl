// MusicCoCa's contrastive text pooler — the step that collapses [n, 768] into the single 768-d
// style embedding the RVQ quantises.
//
// It is NOT a mean over positions. The export attends with a LEARNED query, one per head:
//
//     logits[h, t] = query[h] · K[t, h]          (12 heads x 256, no scale, no logit cap)
//     ctx[h]       = sum_t softmax(logits[h])[t] · V[t, h]
//
// K and V are the 3072-wide projections of the tower's output, viewed as [n, 12, 256]. Getting
// this wrong is quiet: a mean-pool still yields a plausible 768-d vector and plausible style
// tokens, just not the model's.
//
// One work-group per head. `n` is the real token count (<= 128), so the logit row fits in local
// memory and the reduction is a short serial pass rather than a tree.

__kernel void pool_attn_f32(__global const float* K,      // [n, H*HD]
                            __global const float* V,      // [n, H*HD]
                            __global const half* query,   // [H, HD] fp16
                            __global float* ctx,          // [H*HD]
                            const int n,
                            const int HD,
                            const int stride) {           // == H*HD
  const int h = get_group_id(0);
  const int lid = get_local_id(0);
  const int lsz = get_local_size(0);
  __local float p[128];
  __local float inv_sum;

  for (int t = lid; t < n; t += lsz) {
    float s = 0.0f;
    __global const float* krow = K + (size_t)t * stride + (size_t)h * HD;
    for (int d = 0; d < HD; ++d) s += vload_half((size_t)h * HD + d, query) * krow[d];
    p[t] = s;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    float m = -INFINITY;
    for (int t = 0; t < n; ++t) m = fmax(m, p[t]);
    float sum = 0.0f;
    for (int t = 0; t < n; ++t) { p[t] = exp(p[t] - m); sum += p[t]; }
    inv_sum = 1.0f / fmax(sum, 1e-20f);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int d = lid; d < HD; d += lsz) {
    float a = 0.0f;
    for (int t = 0; t < n; ++t) a += p[t] * V[(size_t)t * stride + (size_t)h * HD + d];
    ctx[(size_t)h * HD + d] = a * inv_sum;
  }
}
