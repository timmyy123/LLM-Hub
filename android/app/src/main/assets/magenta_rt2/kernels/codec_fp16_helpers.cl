// Support kernels for the fp16 codec GEMM path.
//
// The GEMM computes  C = (A/s) . W + bias/s , so every operand that meets the fp16 matmul has to be
// pre-divided by the same scale and the result multiplied back. Keeping the bias inside the GEMM
// (as CLBlast's C operand with beta=1) means it must be scaled too, which is why the prefill takes
// the scale rather than the unscale step adding the bias afterwards.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// out[m,n] = bias[n] * inv_scale, as fp16 — the C operand CLBlast accumulates into.
__kernel void bias_prefill_f16(__global const float* bias, __global half* out,
                               const int M, const int N, __global const float* sc) {
  const int i = get_global_id(0);
  if (i >= M*N) return;
  vstore_half(bias[i % N] * sc[1], i, out);
}

// fp32 [N,K] -> fp16 [K,N]. Transposed once per weight and cached: CLBlast can only avoid its
// temporary buffers when the operand needs no transpose.
__kernel void transpose_nk_to_kn_f16(__global const float* src, __global half* dst,
                                     const int N, const int K) {
  const int k = get_global_id(0), n = get_global_id(1);
  if (k >= K || n >= N) return;
  vstore_half(src[(size_t)n * K + k], (size_t)k * N + n, dst);
}

// out_f32[i] = in_f16[i] * scale — undoes the input scaling on the way back to fp32.
__kernel void unscale_f16_to_f32(__global const half* in, __global float* out,
                                 const int n, __global const float* sc) {
  const int i = get_global_id(0);
  if (i >= n) return;
  out[i] = vload_half(i, in) * sc[0];
}

// ── scale selection, entirely on the GPU ─────────────────────────────────────────────────────────
// This used to be one workgroup reducing the whole tensor, followed by a BLOCKING readback so the
// host could compute the scale and pass it as a kernel argument. Both halves were expensive: a late
// cascade block is ~19.6M floats and a single 256-thread workgroup reduces it on ONE compute unit
// while the rest of the GPU idles, and the readback then stalls the pipeline — 85 times per render.
//
// Now: stage 1 reduces across many workgroups into partials, stage 2 combines them and writes the
// scale pair into a 2-float buffer that the gather, the bias prefill and the unscale all read
// directly. The host never sees the value, so nothing blocks.
#define AMWG 256
__kernel __attribute__((reqd_work_group_size(AMWG, 1, 1)))
void absmax_f32_partial(__global const float* x, __global float* part, const int n) {
  const int lid = get_local_id(0);
  const int gid = get_group_id(0), ngroups = get_num_groups(0);
  __local float red[AMWG];
  float m = 0.0f;
  for (int i = gid * AMWG + lid; i < n; i += AMWG * ngroups) m = fmax(m, fabs(x[i]));
  red[lid] = m;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = AMWG/2; s > 0; s >>= 1) {
    if (lid < s) red[lid] = fmax(red[lid], red[lid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) part[gid] = red[0];
}

// sc[0] = scale, sc[1] = 1/scale. One work item; `ngroups` is tiny (64).
//
// The scale CENTRES the block in fp16's exponent window at 2^target rather than pinning its maximum
// to 2^0. fp16 normals run 2^-14..2^15.99, so mapping the peak to 1.0 abandons the fifteen octaves
// above it and pays for that at the bottom, where values fall subnormal and then to zero. Powers of
// two are exact in binary floating point, so the divide-and-multiply-back introduces no rounding.
__kernel void scale_from_partials(__global const float* part, __global float* sc,
                                  const int ngroups, const int target) {
  if (get_global_id(0) != 0) return;
  float m = 0.0f;
  for (int i = 0; i < ngroups; ++i) m = fmax(m, part[i]);
  float s = 1.0f;
  if (m > 0.0f && isfinite(m)) s = exp2(ceil(log2(m)) - (float)target);
  if (!(s > 0.0f) || !isfinite(s)) s = 1.0f;
  sc[0] = s;
  sc[1] = 1.0f / s;
}
