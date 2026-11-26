// LayerNorm, ONE WORKGROUP PER ROW: out = (x-mean)/sqrt(var+eps)*weight + bias.
//
// Same math as layer_norm_f32.cl, which runs an entire row on one work item — one lane of one of
// twelve compute units, since Adreno pins a workgroup to a single CU (80-NB295-11 §3.2.5). This one
// sits at the tail of the depth stack and runs 12x per frame, 1200x per 2 s chunk.
//
// Two reductions (mean, then variance) share the same local buffer. Reduction ORDER differs from
// the serial kernel, so the result can move by a last-ulp amount; the caller checks the audio.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 128

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void layer_norm_wg_f32(__global const float* x, __global const half* w, __global const half* b,
                       __global float* out, const int dim, const float eps){
  const int row = get_group_id(0);
  const int lid = get_local_id(0);
  const size_t base = (size_t)row * dim;
  __local float red[WG];
  __local float mean_l, inv_l;

  float s = 0.0f;
  for (int d = lid; d < dim; d += WG) s += x[base + d];
  red[lid] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = WG / 2; k > 0; k >>= 1) {
    if (lid < k) red[lid] += red[lid + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) mean_l = red[0] / (float)dim;
  barrier(CLK_LOCAL_MEM_FENCE);
  const float mean = mean_l;

  float v = 0.0f;
  for (int d = lid; d < dim; d += WG) { const float t = x[base + d] - mean; v += t * t; }
  red[lid] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = WG / 2; k > 0; k >>= 1) {
    if (lid < k) red[lid] += red[lid + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) inv_l = rsqrt(red[0] / (float)dim + eps);
  barrier(CLK_LOCAL_MEM_FENCE);
  const float inv = inv_l;

  for (int d = lid; d < dim; d += WG)
    out[base + d] = (x[base + d] - mean) * inv * vload_half(d, w) + vload_half(d, b);
}
