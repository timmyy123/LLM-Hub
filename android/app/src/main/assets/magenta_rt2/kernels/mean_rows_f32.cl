// Mean (or scaled sum) over the N rows of a row-major [N,dim] fp32 buffer.
//   out[d] = scale * sum_{q<N} in[q*dim + d]      (scale = 1/N for mean)
// One work-item per output channel d. Keeps the frame-embed reduction on-GPU
// so the AR feedback path never round-trips through host (OPT #5).
__kernel void mean_rows_f32(__global const float* in, __global float* out,
                            const int N, const int dim, const float scale){
  const int d = get_global_id(0); if (d >= dim) return;
  float acc = 0.0f;
  for (int q = 0; q < N; ++q) acc += in[(size_t)q*dim + d];
  out[d] = acc * scale;
}
