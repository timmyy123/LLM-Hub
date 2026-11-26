// Sink attention (cross-attention path), ONE WORKGROUP PER HEAD.
//
// Same math as sink_attention_f32.cl, which is dispatched with global size H (=12) and no local
// size — twelve work items for the whole GPU. Adreno pins a workgroup to one compute unit
// (80-NB295-11 §3.2.5), so twelve items is twelve lanes of a single CU with the other eleven idle.
// This is the temporal cross-attention, 12 calls per frame.
//
// The query attends to the learned sink and to the conditioning source. Upstream's cross-attention
// holds a 41-frame ring of source vectors plus the current one (42 keys); within a chunk the
// conditioning is constant, so all 42 keys are BIT-IDENTICAL (measured: logit spread exactly 0).
// Attending over N identical keys is algebraically the same as attending over one whose logit is
// raised by ln(N), so `log_n` carries the whole ring at zero cost. log_n=0 ⇒ kv = [sink, source],
// which is the self-attention case and the port's pre-2026-09 behaviour.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 64

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void sink_attention_wg_f32(
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
  const int h   = get_group_id(0);
  const int lid = get_local_id(0);
  if (h >= H) return;
  const int   xp_   = xpos_buf[0];
  const int   nsrc_ = (xp_ < 0) ? 1 : ((xp_ + 1 < max_src) ? xp_ + 1 : max_src);
  const float log_n = log((float)nsrc_);
  const int HD = H * D;
  const size_t qb = (size_t)h * D;          // q[h,*] in q
  const size_t kb = qb;                     // k[h,*] in kv
  const size_t vb = (size_t)HD + qb;        // v[h,*] in kv

  // r_softplus_0 = 1/ln2 normalises so per_dim_scale=0 -> scale=query_scale (attention.py:38).
  const float r_softplus_0 = 1.442695041f;
  __local float rs[WG], rc[WG];
  __local float w0_l, w1_l;

  float s_sink = 0.0f, s_cur = 0.0f;
  for (int d = lid; d < D; d += WG) {
    const float qd = q[qb + d];
    const float sv = r_softplus_0 * inv_sqrt_d * log1p(exp(vload_half(d, per_dim_scale)));
    s_sink += qd * vload_half(qb + d, sink_k);
    s_cur  += (qd * sv) * kv[kb + d];
  }
  rs[lid] = s_sink; rc[lid] = s_cur;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = WG / 2; k > 0; k >>= 1) {
    if (lid < k) { rs[lid] += rs[lid + k]; rc[lid] += rc[lid + k]; }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    const float a = rs[0], b = rc[0] + log_n;
    const float mx = a > b ? a : b;
    const float e0 = exp(a - mx), e1 = exp(b - mx);
    const float z = e0 + e1;
    w0_l = e0 / z; w1_l = e1 / z;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float w0 = w0_l, w1 = w1_l;

  for (int d = lid; d < D; d += WG)
    out[qb + d] = w0 * vload_half(qb + d, sink_v) + w1 * kv[vb + d];
}
