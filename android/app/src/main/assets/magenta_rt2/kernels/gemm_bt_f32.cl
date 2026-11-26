// LDS-tiled GEMM: C[M,N] = A[M,K] @ B[N,K].T + bias[N]. 16x16 tile, K reused 16x from local memory.
// A=im2col [M,K], B=conv weight [Cout=N, K=kH*kW*Cin] (contiguous). gws=(ceil(N/16)*16, ceil(M/16)*16), lws=16x16.
#define TS 16
__kernel void gemm_bt_f32(__global const float* A, __global const float* B, __global const float* bias,
                          __global float* C, const int M, const int N, const int K){
  __local float As[TS][TS];
  __local float Bs[TS][TS];
  const int lx=get_local_id(0), ly=get_local_id(1);
  const int n=get_group_id(0)*TS+lx, m=get_group_id(1)*TS+ly;
  float acc=0.0f;
  for(int k0=0;k0<K;k0+=TS){
    As[ly][lx] = (m<M && k0+lx<K) ? A[(size_t)m*K+k0+lx] : 0.0f;  // A row m, K within chunk
    Bs[lx][ly] = (n<N && k0+ly<K) ? B[(size_t)n*K+k0+ly] : 0.0f;  // B row n (=Cout), K within chunk
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(int kk=0;kk<TS;kk++) acc += As[ly][kk]*Bs[lx][kk];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(m<M && n<N) C[(size_t)m*N+n] = acc + bias[n];
}
