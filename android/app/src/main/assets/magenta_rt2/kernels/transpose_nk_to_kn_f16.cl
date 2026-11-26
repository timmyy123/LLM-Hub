// Transpose an fp16 weight matrix [N,K] -> [K,N], once per tensor, cached for the process lifetime.
//
// The AR GEMV reads W row-major per output ([N,K]), which forces a workgroup-wide tree reduction to
// combine each thread's partial dot product. In K-major ([K,N]) one thread owns one output and walks
// k by itself: no barriers, no local memory, and adjacent threads read adjacent halves so a wave
// still issues one full-width transaction. See the bonsai8b port, where the same K-major
// thread-per-row shape was the winning GEMV on Adreno.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void transpose_nk_to_kn_f16(__global const half* src,   // [N, K]
                                     __global half* dst,          // [K, N]
                                     const int N,
                                     const int K) {
  const int k = get_global_id(0);
  const int n = get_global_id(1);
  if (k >= K || n >= N) return;
  dst[(size_t)k * N + n] = src[(size_t)n * K + k];
}
