// RMSNormalization (sequence_layers): out = x / sqrt(mean(x^2) + eps) * scale.
// Activations fp32; scale stored fp16 (vload_half). One work-item per row.
// Reference: magenta_rt RMSNormalization (epsilon=1e-6, use_scale=True).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rms_norm_f32(
    __global const float* x,      // [rows, dim] fp32
    __global const half*  scale,  // [dim] fp16
    __global float*       out,     // [rows, dim] fp32
    const int dim,
    const float eps) {
  const int row = get_global_id(0);
  const size_t base = (size_t)row * dim;
  float ss = 0.0f;
  for (int d = 0; d < dim; d++) { float v = x[base + d]; ss += v * v; }
  const float inv = rsqrt(ss / (float)dim + eps);
  for (int d = 0; d < dim; d++) {
    out[base + d] = x[base + d] * inv * vload_half(d, scale);
  }
}
