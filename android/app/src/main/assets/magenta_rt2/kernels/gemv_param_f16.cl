// Parameterized wave-stride GEMV for sweeping. -D WGSZ=<n> -D NOUT=<n> -D VW=<4|8>.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef WGSZ
#define WGSZ 64
#endif
#ifndef NOUT
#define NOUT 4
#endif
__kernel void gemv_param_f16(__global const float* x, __global const half* W,
                             __global float* out, const int in_dim, const int out_dim){
  __local float ls[WGSZ*NOUT];
  const int tid=get_local_id(1), wg=get_group_id(1), n0=wg*NOUT;
  const int in4=in_dim>>2;
  float acc[NOUT];
  #pragma unroll
  for(int u=0;u<NOUT;u++) acc[u]=0.0f;
  for(int j=tid;j<in4;j+=WGSZ){
    float4 xv=vload4(j,x);
    #pragma unroll
    for(int u=0;u<NOUT;u++) acc[u]+=(n0+u<out_dim)?dot(xv,vload_half4(j,W+(size_t)(n0+u)*in_dim)):0.0f;
  }
  #pragma unroll
  for(int u=0;u<NOUT;u++) ls[tid+u*WGSZ]=acc[u];
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int s=WGSZ/2;s>0;s>>=1){ if(tid<s){ for(int u=0;u<NOUT;u++) ls[tid+u*WGSZ]+=ls[tid+s+u*WGSZ]; } barrier(CLK_LOCAL_MEM_FENCE); }
  if(tid<NOUT && n0+tid<out_dim) out[n0+tid]=ls[tid*WGSZ];
}
