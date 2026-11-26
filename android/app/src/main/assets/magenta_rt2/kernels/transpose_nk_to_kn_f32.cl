// Transpose a conv weight matrix from [N,K] to [K,N], once, so CLBlast never has to.
//
// conv_clblast called CLBlastSgemm with b_transpose=TransposeYes. CLBlast can only skip its
// temporary buffers when `!do_transpose` (NoTempBuffer in xgemm.hpp), so that flag forced it to
// allocate a temp and transpose the ENTIRE weight matrix on every conv call — for weights that are
// constant for the life of the process. Measured: 10.9 s inside CLBlast across 120 calls while the
// actual matmul was 61 ms.
//
// Transposing once at first use and passing TransposeNo removes that work permanently.
__kernel void transpose_nk_to_kn_f32(__global const float* src,   // [N, K]
                                     __global float* dst,          // [K, N]
                                     const int N,
                                     const int K) {
  const int k = get_global_id(0);
  const int n = get_global_id(1);
  if (k >= K || n >= N) return;
  dst[(size_t)k * N + n] = src[(size_t)n * K + k];
}
