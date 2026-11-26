// Wave-stride GEMV + bias, 8 outputs/WG. Dispatch gws={rows, ceil(out/8)*64}, lws={1,64}.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 64
#define NOUT 8
__kernel void linear_bias_f32(__global const float* x, __global const half* W, __global const half* bias,
                              __global float* out, const int in_dim, const int out_dim){
  __local float ls[WG*NOUT];
  const int row=get_global_id(0), tid=get_local_id(1), wg=get_group_id(1), n0=wg*NOUT;
  const size_t xb=(size_t)row*in_dim; const int in4=in_dim>>2;
  float acc[NOUT];
  #pragma unroll
  for(int u=0;u<NOUT;u++) acc[u]=0.0f;
  for(int j=tid;j<in4;j+=WG){
    float4 xv=vload4(j,x+xb);
    #pragma unroll
    for(int u=0;u<NOUT;u++) acc[u]+=(n0+u<out_dim)?dot(xv,vload_half4(j,W+(size_t)(n0+u)*in_dim)):0.0f;
  }
  #pragma unroll
  for(int u=0;u<NOUT;u++) ls[tid+u*WG]=acc[u];
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int s=WG/2;s>0;s>>=1){ if(tid<s){ for(int u=0;u<NOUT;u++) ls[tid+u*WG]+=ls[tid+s+u*WG]; } barrier(CLK_LOCAL_MEM_FENCE); }
  if(tid<NOUT && n0+tid<out_dim) out[(size_t)row*out_dim+n0+tid]=ls[tid*WG]+vload_half(n0+tid,bias);
}
