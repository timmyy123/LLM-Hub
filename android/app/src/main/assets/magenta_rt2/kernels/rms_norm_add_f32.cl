// Fused RMSNorm + residual-add (AR layer-fusion): out = residual + rmsnorm(x)*scale.
// Every transformer block ends `out = x_block + rms2(proj)`; fusing the post-norm and the residual
// add into one kernel removes one enqueue per block (~86/frame) — the AR is host-issue-bound, so
// fewer enqueues = less wall. Same math as rms_norm_f32 followed by add_f32 (bit-identical).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rms_norm_add_f32(
    __global const float* x,         // [rows, dim] fp32 — norm input (e.g. proj / ff)
    __global const half*  scale,     // [dim] fp16
    __global const float* residual,  // [rows, dim] fp32 — the pre-block input to add
    __global float*       out,        // [rows, dim] fp32
    const int dim,
    const float eps) {
  const int row = get_global_id(0);
  const size_t base = (size_t)row * dim;
  float ss = 0.0f;
  for (int d = 0; d < dim; d++) { float v = x[base + d]; ss += v * v; }
  const float inv = rsqrt(ss / (float)dim + eps);
  for (int d = 0; d < dim; d++) {
    out[base + d] = x[base + d] * inv * vload_half(d, scale) + residual[base + d];
  }
}
