// Residual vector quantisation — the MusicCoCa embedding → 12 style tokens.
//
// One stage: find the codebook row nearest the residual, emit its index, subtract it, hand the new
// residual to the next stage. Twelve stages give the twelve style tokens the LLM's cross-attention
// consumes.
//
// Nearest by squared L2, expanded so the shared |r|^2 term drops out:
//     argmin_n |r - c_n|^2  ==  argmin_n ( |c_n|^2 - 2 r·c_n )
//
// codebook [N, D] fp16, residual [D] fp32. One work-group; each thread scans a stride of rows and
// keeps its own best, then a local-memory reduction picks the winner. Ties resolve to the LOWEST
// index, matching argmin semantics in the reference — a tie broken the other way would silently
// produce a different (and still plausible-sounding) style.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rvq_quantize_f32(__global const float* residual,  // [D]
                               __global const half* codebook,   // [N, D]
                               __global int* out_index,         // [1]
                               __global float* out_residual,    // [D] = residual - codebook[idx]
                               const int N,
                               const int D) {
  const int lid = get_local_id(0);
  const int lsz = get_local_size(0);
  __local float best_d[256];
  __local int   best_i[256];

  float bd = INFINITY;
  int bi = 0;
  for (int n = lid; n < N; n += lsz) {
    __global const half* c = codebook + (size_t)n * D;
    float dot = 0.0f, nrm = 0.0f;
    for (int d = 0; d < D; ++d) {
      const float cv = vload_half(d, c);
      dot += residual[d] * cv;
      nrm += cv * cv;
    }
    const float dist = nrm - 2.0f * dot;      // |r|^2 omitted: constant across n
    if (dist < bd || (dist == bd && n < bi)) { bd = dist; bi = n; }
  }
  best_d[lid] = bd; best_i[lid] = bi;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsz / 2; s > 0; s >>= 1) {
    if (lid < s) {
      const float od = best_d[lid + s];
      const int oi = best_i[lid + s];
      if (od < best_d[lid] || (od == best_d[lid] && oi < best_i[lid])) {
        best_d[lid] = od; best_i[lid] = oi;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const int idx = best_i[0];
  if (lid == 0) out_index[0] = idx;
  barrier(CLK_LOCAL_MEM_FENCE);

  __global const half* c = codebook + (size_t)idx * D;
  for (int d = lid; d < D; d += lsz) out_residual[d] = residual[d] - vload_half(d, c);
}
