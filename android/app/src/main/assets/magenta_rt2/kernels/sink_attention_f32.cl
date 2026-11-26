// SinkAttention core (DeferredLocalDotProductSelfAttention), single query frame.
// Reference: magenta_rt/mlx/attention.py LocalDotProductSelfAttention.__call__
//
// kv positions for the first/standalone frame = [sink, current_self] (kv_len=2).
// Per head h (H heads, D dims/head):
//   scale_vec[d] = inv_sqrt_d * softplus(per_dim_scale[d])
//   s_sink = sum_d  q[h,d] * sink_k[h,d]              (sink_k pre-divided by scale ⇒ unscaled q)
//   s_cur  = sum_d (q[h,d]*scale_vec[d]) * k[h,d]
//   softmax([s_sink, s_cur]) → (w0, w1)
//   out[h,d] = w0*sink_v[h,d] + w1*v[h,d]
//
// log_n raises the source logit by ln(N) to stand in for N identical source keys — the
// cross-attention path's 42-deep conditioning ring (see sink_attention_wg_f32.cl). Self-attention
// passes 0.0, which leaves the math exactly as it was.
//
// qkv layout: [q(H*D) | k(H*D) | v(H*D)] fp32 (output of qkv_proj linear).
// sink_k/sink_v/per_dim_scale fp16 (vload_half). out [H*D] fp32. One work-item per head.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void sink_attention_f32(
    // q and kv arrive as the SEPARATE buffers qkv_proj already produced, instead of being packed
    // into one [3*H*D] block by two copy dispatches first. The packing was 2 dispatches per
    // cross-attention call (24/frame) moving 0.05 ms of data — i.e. ~500:1 host-launch cost against
    // actual work. Unpacked costs nothing: it is the same values at a different offset.
    __global const float* q,            // [H*D]
    __global const float* kv,           // [2*H*D] — k then v
    __global const half*  sink_k,       // [H*D]
    __global const half*  sink_v,       // [H*D]
    __global const half*  per_dim_scale,// [D]
    __global float*       out,          // [H*D]
    const int H,
    const int D,
    const float inv_sqrt_d,
    // ln(N) is derived ON DEVICE from xpos rather than passed as a host float. N grows every
    // frame until it saturates at max_src, so a host-set log_n is a per-frame value baked
    // into a kernel argument — and a recording pins argument values at capture, which froze
    // the conditioning attenuation at the captured frame's ln(2) for every replayed frame.
    __global const int* xpos_buf,   // absolute cross-attention position; <0 means "no counter"
    const int max_src) {            // ring depth (kMaxSrc); N = min(xpos+1, max_src)
  const int h = get_global_id(0);
  if (h >= H) return;
  const int   xp_   = xpos_buf[0];
  const int   nsrc_ = (xp_ < 0) ? 1 : ((xp_ + 1 < max_src) ? xp_ + 1 : max_src);
  const float log_n = log((float)nsrc_);
  const int HD = H * D;
  const size_t qb = (size_t)h * D;          // q[h,*] in q
  const size_t kb = qb;                     // k[h,*] in kv
  const size_t vb = (size_t)HD + qb;        // v[h,*] in kv

  // scale_vec = r_softplus_0 * query_scale * softplus(per_dim_scale);
  // r_softplus_0 = 1/ln2 normalizes so per_dim_scale=0 → scale=query_scale (attention.py:38).
  const float r_softplus_0 = 1.442695041f;
  float s_sink = 0.0f, s_cur = 0.0f;
  for (int d = 0; d < D; d++) {
    const float qd = q[qb + d];
    const float sv = r_softplus_0 * inv_sqrt_d * log1p(exp(vload_half(d, per_dim_scale)));
    s_sink += qd * vload_half(qb + d, sink_k);
    s_cur  += (qd * sv) * kv[kb + d];
  }
  s_cur += log_n;
  const float mx = s_sink > s_cur ? s_sink : s_cur;
  const float e0 = exp(s_sink - mx), e1 = exp(s_cur - mx);
  const float z = e0 + e1;
  const float w0 = e0 / z, w1 = e1 / z;
  for (int d = 0; d < D; d++)
    out[qb + d] = w0 * vload_half(qb + d, sink_v) + w1 * kv[vb + d];
}
