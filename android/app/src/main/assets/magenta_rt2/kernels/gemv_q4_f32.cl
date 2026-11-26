// Block-32 symmetric Q4 GEMV — the DRAM-bound AR decode path at 4 bits.
//
// out = x @ W.T   with   x[rows, K] fp32,  W q4-packed [N, K/2],  scales fp16 [N, K/32]
//
// Layout is byte-identical to scripts/quantize_q4.py:
//   packed byte b holds two weights — low nibble = element 2i, high nibble = element 2i+1,
//   each stored as q+8 in {1..15}; dequant is (stored - 8) * scale[row][k/32].
//
// One uchar4 load = 4 bytes = 8 consecutive weights, which is exactly two float4 of
// activations, and 8 divides the 32-element scale block, so a group never straddles two
// scales. Same WG=64 / NOUT=8 shape as linear_f32_w16 (the tuned fp16 GEMV) so the
// dispatch and the local-memory reduction are unchanged — only the operand width differs.
//
// Dispatch: gws = {rows, ceil(out_dim/8)*64}, lws = {1, 64}.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 64
#define NOUT 8

#define Q4_UNPACK(p, w0, w1) do {                                                     \
    w0 = (float4)((float)((p).x & 0xF), (float)((p).x >> 4),                          \
                  (float)((p).y & 0xF), (float)((p).y >> 4)) - (float4)(8.0f);        \
    w1 = (float4)((float)((p).z & 0xF), (float)((p).z >> 4),                          \
                  (float)((p).w & 0xF), (float)((p).w >> 4)) - (float4)(8.0f);        \
} while (0)

__kernel void gemv_q4_f32(__global const float* x, __global const uchar* W,
                          __global const half* S, __global float* out,
                          const int in_dim, const int out_dim) {
  __local float ls[WG*NOUT];
  const int row = get_global_id(0), tid = get_local_id(1), wg = get_group_id(1), n0 = wg*NOUT;
  const size_t xb = (size_t)row*in_dim;
  const int in8 = in_dim >> 3, wstride = in_dim >> 1, sstride = in_dim >> 5;
  float acc[NOUT];
  #pragma unroll
  for (int u = 0; u < NOUT; u++) acc[u] = 0.0f;
  for (int j = tid; j < in8; j += WG) {
    const float4 x0 = vload4(2*j,   x+xb);
    const float4 x1 = vload4(2*j+1, x+xb);
    const int blk = j >> 2;
    #pragma unroll
    for (int u = 0; u < NOUT; u++) {
      if (n0+u < out_dim) {
        const uchar4 p = vload4(j, W + (size_t)(n0+u)*wstride);
        const float sc = vload_half(blk, S + (size_t)(n0+u)*sstride);
        float4 w0, w1; Q4_UNPACK(p, w0, w1);
        acc[u] += sc * (dot(x0, w0) + dot(x1, w1));
      }
    }
  }
  #pragma unroll
  for (int u = 0; u < NOUT; u++) ls[tid+u*WG] = acc[u];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG/2; s > 0; s >>= 1) {
    if (tid < s) { for (int u = 0; u < NOUT; u++) ls[tid+u*WG] += ls[tid+s+u*WG]; }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid < NOUT && n0+tid < out_dim) out[(size_t)row*out_dim+n0+tid] = ls[tid*WG];
}

// Same, plus a per-output bias (MLP dense layers and the depth adapter carry one).
__kernel void gemv_q4_bias_f32(__global const float* x, __global const uchar* W,
                               __global const half* S, __global const float* bias,
                               __global float* out, const int in_dim, const int out_dim) {
  __local float ls[WG*NOUT];
  const int row = get_global_id(0), tid = get_local_id(1), wg = get_group_id(1), n0 = wg*NOUT;
  const size_t xb = (size_t)row*in_dim;
  const int in8 = in_dim >> 3, wstride = in_dim >> 1, sstride = in_dim >> 5;
  float acc[NOUT];
  #pragma unroll
  for (int u = 0; u < NOUT; u++) acc[u] = 0.0f;
  for (int j = tid; j < in8; j += WG) {
    const float4 x0 = vload4(2*j,   x+xb);
    const float4 x1 = vload4(2*j+1, x+xb);
    const int blk = j >> 2;
    #pragma unroll
    for (int u = 0; u < NOUT; u++) {
      if (n0+u < out_dim) {
        const uchar4 p = vload4(j, W + (size_t)(n0+u)*wstride);
        const float sc = vload_half(blk, S + (size_t)(n0+u)*sstride);
        float4 w0, w1; Q4_UNPACK(p, w0, w1);
        acc[u] += sc * (dot(x0, w0) + dot(x1, w1));
      }
    }
  }
  #pragma unroll
  for (int u = 0; u < NOUT; u++) ls[tid+u*WG] = acc[u];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = WG/2; s > 0; s >>= 1) {
    if (tid < s) { for (int u = 0; u < NOUT; u++) ls[tid+u*WG] += ls[tid+s+u*WG]; }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid < NOUT && n0+tid < out_dim) out[(size_t)row*out_dim+n0+tid] = ls[tid*WG] + bias[n0+tid];
}
