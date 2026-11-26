// Reference: /Users/alazarshenkute/.nnopt/repos/magenta-realtime (MLX MagentaRT2)
// Reference: /Users/alazarshenkute/Projects/nnopt/magenta-realtime-2-android-opencl/reference/forward_graph.json (QuantizedEmbedding nodes)
//
// Simple embedding gather for debugging parity: out[t, d] = table[id[t], d].
// Table is stored as fp32 in weights/model.fp16.bin? (meta declares dtype).
// Kernel assumes table stored as float; if table is storage_t, adjust later.

#define CL_TARGET_OPENCL_VERSION 120

__kernel void embedding_gather_f32(
    __global const float* table,   // [vocab, dim]
    __global const int* ids,       // [n_ids]
    __global float* out,           // [n_ids, dim]
    const int dim,
    const int n_ids,
    const int vocab) {
  const int t = get_global_id(0);
  const int d = get_global_id(1);
  if (t >= n_ids || d >= dim) return;
  int id = ids[t];
  if (id < 0) id = 0;
  if (id >= vocab) id = vocab - 1;
  out[t * dim + d] = table[id * dim + d];
}
