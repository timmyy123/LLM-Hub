// fp16-storage embedding gather: out[t, d] = table[id[t], d] * sqrt(dim).
// The decoder embedder applies a sqrt(embedding_dim) scale (sl.Embedding _scale,
// default = sqrt(dim) = 32 for dim=1024). Cosine-invariant (so per-node cosine
// gates miss it) but the depth-loop adapter is scale-sensitive — omitting it
// breaks every codebook after the first. Weights stored fp16 (vload_half); out fp32.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void embedding_gather_f16(
    __global const half* table,   // [vocab, dim] fp16
    __global const int*  ids,     // [n_ids] int32
    __global float*      out,      // [n_ids, dim] fp32
    const int dim,
    const int n_ids,
    const int vocab) {
  const int t = get_global_id(0);
  const int d = get_global_id(1);
  if (t >= n_ids || d >= dim) return;
  int id = ids[t];
  if (id < 0) id = 0;
  if (id >= vocab) id = vocab - 1;
  out[t * dim + d] = vload_half((size_t)id * dim + d, table) * sqrt((float)dim);
}
