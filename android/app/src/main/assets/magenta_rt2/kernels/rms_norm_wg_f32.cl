// RMSNormalization, ONE WORKGROUP PER ROW: out = x / sqrt(mean(x^2) + eps) * scale.
//
// Same math as rms_norm_f32.cl, which runs the whole row on a single work item — one lane of one
// of twelve compute units, because Adreno cannot split a workgroup across CUs (80-NB295-11 §3.2.5).
// This is the pre-norm at the head of every attention and MLP block: 84 of them per frame, 8400 per
// 2 s chunk, and it was enqueued with a raw clEnqueueNDRangeKernel rather than the profiling
// wrapper, so it never appeared in any profile — invisible AND serial.
//
// The reduction is a local-memory tree, so the sum-of-squares ORDER differs from the serial kernel
// and the normalisation constant can move by an ulp. In an autoregressive model that can flip an
// argmax, so the caller checks the audio is unchanged.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 128

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void rms_norm_wg_f32(
    __global const float* x,      // [rows, dim] fp32
    __global const half*  scale,  // [dim] fp16
    __global float*       out,    // [rows, dim] fp32
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

  for (int d = lid; d < dim; d += WG) out[base + d] = x[base + d] * inv * vload_half(d, scale);
}
