// GEMV with a SUBGROUP reduction instead of a local-memory tree.
//
// The default kernel (linear_f32_w16.cl) has 64 threads stride the input, then combines their
// partials through __local with a 6-step tree: six barriers and 64*NOUT floats of LDS traffic. For
// the AR's shapes that reduction dominates — at in_dim 768 the accumulate loop runs only THREE
// times before those six barriers. gemv is 75.6% of AR GPU time, so the barriers are the target.
//
// sub_group_reduce_add does the same combine inside the wave, in registers: no LDS, no barrier.
// The Adreno 840 exposes cl_khr_subgroups and cl_qcom_subgroup_shuffle but NOT
// cl_qcom_reqd_sub_group_size, so the wave size cannot be pinned and must not be assumed. When the
// workgroup happens to span more than one subgroup we still need one combine across them — that is
// a single barrier over `nsg` values, not six over 64. `nsg` is uniform across the workgroup, so
// branching on it around a barrier is well-defined.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#define WG 64
#define NOUT 8
#define MAXSG 8   // WG/8 — the smallest wave this kernel will see is 8 lanes
__kernel void linear_f32_w16_sg(__global const float* x, __global const half* W,
                                __global float* out, const int in_dim, const int out_dim){
  __local float ls[MAXSG*NOUT];
  const int row=get_global_id(0), tid=get_local_id(1), wg=get_group_id(1), n0=wg*NOUT;
  const size_t xb=(size_t)row*in_dim; const int in4=in_dim>>2;
  float acc[NOUT];
  #pragma unroll
  for(int u=0;u<NOUT;u++) acc[u]=0.0f;
  // Identical gather to the default kernel: adjacent lanes read adjacent half4, so the wave still
  // issues full-width transactions. Only the combine below changes.
  for(int j=tid;j<in4;j+=WG){
    float4 xv=vload4(j,x+xb);
    #pragma unroll
    for(int u=0;u<NOUT;u++) acc[u]+=(n0+u<out_dim)?dot(xv,vload_half4(j,W+(size_t)(n0+u)*in_dim)):0.0f;
  }

  float part[NOUT];
  #pragma unroll
  for(int u=0;u<NOUT;u++) part[u]=sub_group_reduce_add(acc[u]);

  const int nsg   = get_num_sub_groups();
  const int sgid  = get_sub_group_id();
  const int sglid = get_sub_group_local_id();

  if (nsg == 1) {
    // The whole workgroup is one wave: every lane already holds the finished sums. One lane writes
    // the NOUT results — a handful of floats per workgroup, negligible next to the weight reads.
    if (sglid == 0) {
      #pragma unroll
      for(int u=0;u<NOUT;u++) if(n0+u<out_dim) out[(size_t)row*out_dim+n0+u]=part[u];
    }
  } else {
    if (sglid == 0) {
      #pragma unroll
      for(int u=0;u<NOUT;u++) ls[sgid*NOUT+u]=part[u];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < NOUT && n0+tid < out_dim) {
      float s=0.0f;
      for(int g=0;g<nsg;g++) s+=ls[g*NOUT+tid];
      out[(size_t)row*out_dim+n0+tid]=s;
    }
  }
}
