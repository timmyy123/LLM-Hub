// MusicCoCa input layer: token embedding + learned positional table, in one pass.
//
// Both tables are fp16 on device and the activations are fp32, so this cannot be an
// embedding_gather followed by add_f32 — that kernel adds two fp32 buffers and would read the
// fp16 positional table as garbage. Fusing them also saves a full [n,768] round-trip.
//
//   out[t, d] = embed[ids[t], d] + pos[t, d]
//
// `n` is the REAL token count, not the exported 128: padded positions are masked out of every
// attention downstream, so running the tower at the prompt's own length is bit-identical.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void embed_add_pos_f16(__global const half* table,  // [vocab, dim]
                                __global const half* pos,    // [max_seq, dim]
                                __global const int* ids,     // [n]
                                __global float* out,         // [n, dim]
                                const int dim,
                                const int n,
                                const int vocab) {
  const int t = get_global_id(0);
  const int d = get_global_id(1);
  if (t >= n || d >= dim) return;
  int id = ids[t];
  if (id < 0 || id >= vocab) id = 0;
  out[(size_t)t * dim + d] =
      vload_half((size_t)id * dim + d, table) + vload_half((size_t)t * dim + d, pos);
}
