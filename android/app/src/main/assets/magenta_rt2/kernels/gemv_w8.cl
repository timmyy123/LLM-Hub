// int8-weight GEMV (feasibility probe): out[n] = rowscale[n] * sum_k x[k]*W8[n,k].
// W8 int8 (per-output-row symmetric quant), x fp32. Reads 4 int8/load (char4) → HALF the weight
// DRAM of the fp16 gemv (linear_f32_w16 reads vload_half4 = 8 bytes). Same wave-stride/NOUT layout
// so the only variable vs the fp16 kernel is weight byte-width — isolates "does halving weight
// bandwidth speed up the AR gemv, or is it issue/occupancy-bound?".
#define WG 64
#define NOUT 8
__kernel void gemv_w8(__global const float* x, __global const char* W8, __global const float* rowscale,
                      __global float* out, const int in_dim, const int out_dim){
  __local float ls[WG*NOUT];
  const int tid=get_local_id(1), wg=get_group_id(1), n0=wg*NOUT;
  const int in4=in_dim>>2;
  float acc[NOUT];
  #pragma unroll
  for(int u=0;u<NOUT;u++) acc[u]=0.0f;
  for(int j=tid;j<in4;j+=WG){
    float4 xv=vload4(j,x);
    #pragma unroll
    for(int u=0;u<NOUT;u++){
      char4 w=vload4(j, W8+(size_t)(n0+u)*in_dim);
      acc[u]+=(n0+u<out_dim)? dot(xv, convert_float4(w)) : 0.0f;
    }
  }
  #pragma unroll
  for(int u=0;u<NOUT;u++) ls[tid+u*WG]=acc[u];
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int s=WG/2;s>0;s>>=1){ if(tid<s){ for(int u=0;u<NOUT;u++) ls[tid+u*WG]+=ls[tid+s+u*WG]; } barrier(CLK_LOCAL_MEM_FENCE); }
  if(tid<NOUT && n0+tid<out_dim) out[(size_t)n0+tid]=ls[tid*WG]*rowscale[n0+tid];
}
