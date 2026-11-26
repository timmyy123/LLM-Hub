// Fused RMSNorm + residual-add, ONE WORKGROUP PER ROW.
//
// Same math as rms_norm_add_f32.cl. The difference is the launch: that kernel is dispatched with a
// global size of 1, so a single work item walks all 768–1024 elements twice (sum of squares, then
// normalise) while 63 of its wave's 64 lanes and 11 of the GPU's 12 compute units sit idle. It
// measured 384 us per call for ~6 KB of traffic, and it is called 84 times per frame.
//
// Adreno assigns a workgroup to ONE compute unit and cannot split it across several
// (80-NB295-11 §3.2.5), so a 1-item dispatch can never be more than one lane of one CU no matter
// how large the GPU is. Parallelising within the row is the only way to use the hardware here.
//
// Reduction is a local-memory tree. NOTE this changes the summation ORDER of the sum of squares
// versus the serial version, so the normalisation constant can differ in the last fp32 ulp — which
// in an autoregressive model can flip an argmax. The caller verifies the audio is unchanged.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 128

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void rms_norm_add_wg_f32(
    __global const float* x,         // [rows, dim] fp32 — norm input
    __global const half*  scale,     // [dim] fp16
    __global const float* residual,  // [rows, dim] fp32 — the pre-block input to add
    __global float*       out,       // [rows, dim] fp32
    const int dim,
    const float eps) {
  const int row = get_group_id(0);
  const int lid = get_local_id(0);
  const size_t base = (size_t)row * dim;
  __local float red[WG];

  float ss = 0.0f;
  for (int d = lid; d < dim; d += WG) { const float v = x[base + d]; ss += v * v; }
  red[lid] = ss;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG / 2; s > 0; s >>= 1) {
    if (lid < s) red[lid] += red[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = rsqrt(red[0] / (float)dim + eps);

  for (int d = lid; d < dim; d += WG) {
    out[base + d] = x[base + d] * inv * vload_half(d, scale) + residual[base + d];
  }
}
