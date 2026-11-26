// Sliced wave-stride GEMV+bias: computes ONLY output rows [row_off, row_off+count) of W, into out[0..count).
// out[i] = x @ W[row_off+i, :].T + bias[row_off+i]. 8 outputs/WG. gws={1, ceil(count/8)*64}, lws={1,64}.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 64
#define NOUT 8
__kernel void linear_slice_bias_f32(__global const float* x, __global const half* W, __global const half* bias,
                                    __global float* out, const int in_dim, const int row_off, const int count){
  __local float ls[WG*NOUT];
  const int tid=get_local_id(1), wg=get_group_id(1), i0=wg*NOUT;
  const int in4=in_dim>>2;
  float acc[NOUT];
  #pragma unroll
  for(int u=0;u<NOUT;u++) acc[u]=0.0f;
  for(int j=tid;j<in4;j+=WG){
    float4 xv=vload4(j,x);
    #pragma unroll
    for(int u=0;u<NOUT;u++) acc[u]+=(i0+u<count)?dot(xv,vload_half4(j,W+(size_t)(row_off+i0+u)*in_dim)):0.0f;
  }
  #pragma unroll
  for(int u=0;u<NOUT;u++) ls[tid+u*WG]=acc[u];
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int s=WG/2;s>0;s>>=1){ if(tid<s){ for(int u=0;u<NOUT;u++) ls[tid+u*WG]+=ls[tid+s+u*WG]; } barrier(CLK_LOCAL_MEM_FENCE); }
  if(tid<NOUT && i0+tid<count) out[i0+tid]=ls[tid*WG]+vload_half(row_off+i0+tid,bias);
}
