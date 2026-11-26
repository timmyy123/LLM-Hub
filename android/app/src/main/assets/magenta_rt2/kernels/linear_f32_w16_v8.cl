// Wave-stride GEMV with 128-BIT weight loads. Same math as linear_f32_w16.cl, different memory shape.
//
// The existing kernel loads weights with vload_half4 — 4 halves = 64 bits per transaction. The
// Adreno programming guide is explicit that this leaves half the bandwidth on the table:
//
//   "Adreno GPUs support up to 128-bit read/write of global/local memory and image per load/store
//    transaction. To maximize the memory load efficiency, each work item ideally should use the
//    vectorized load/store functions ... This is helpful for memory-bound use cases."
//   — 80-NB295-11 Rev. C, §6.3 Vectorized load/store
//
// The AR is exactly that case: weights are essentially all of its traffic (95 GB per 2 s chunk),
// so the weight load is the transaction that matters. vload_half8 makes it 128-bit.
//
// The activation is read as two float4s rather than one float8, because 8 floats is 256 bits —
// past the per-transaction maximum — and would be split anyway.
//
// Requires in_dim % 8 == 0. Every AR shape here is a multiple of 64, so that holds; the caller
// falls back to the half4 kernel if it ever does not.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 64
#define NOUT 8
__kernel void linear_f32_w16_v8(__global const float* x, __global const half* W,
                                __global float* out, const int in_dim, const int out_dim){
  __local float ls[WG*NOUT];
  const int row=get_global_id(0), tid=get_local_id(1), wg=get_group_id(1), n0=wg*NOUT;
  const size_t xb=(size_t)row*in_dim; const int in8=in_dim>>3;
  float acc[NOUT];
  #pragma unroll
  for(int u=0;u<NOUT;u++) acc[u]=0.0f;
  for(int j=tid;j<in8;j+=WG){
    const float4 xa=vload4(2*j,   x+xb);
    const float4 xc=vload4(2*j+1, x+xb);
    #pragma unroll
    for(int u=0;u<NOUT;u++){
      if(n0+u<out_dim){
        const float8 w=vload_half8(j, W+(size_t)(n0+u)*in_dim);   // 128-bit transaction
        acc[u]+=dot(xa,w.lo)+dot(xc,w.hi);
      }
    }
  }
  #pragma unroll
  for(int u=0;u<NOUT;u++) ls[tid+u*WG]=acc[u];
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int s=WG/2;s>0;s>>=1){ if(tid<s){ for(int u=0;u<NOUT;u++) ls[tid+u*WG]+=ls[tid+s+u*WG]; } barrier(CLK_LOCAL_MEM_FENCE); }
  if(tid<NOUT && n0+tid<out_dim) out[(size_t)row*out_dim+n0+tid]=ls[tid*WG];
}
