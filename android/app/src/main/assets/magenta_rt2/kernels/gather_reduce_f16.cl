// Gather rows of a fp16 embedding table by (token+offset) and reduce (sum/mean) over N rows.
//   idx_n = tokens[tok_start+n] + offsets[n];  out[d] = scale * sum_n table[idx_n*dim + d]
// table [V,dim] fp16 ; tokens,offsets int32 ; out [dim] fp32. scale=1 (sum) or 1/N (mean).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gather_reduce_f16(__global const half* table, __global const int* tokens,
                                const int tok_start, __global const int* offsets,
                                __global float* out, const int N, const int dim, const float scale){
  const int d=get_global_id(0); if(d>=dim) return;
  float acc=0.0f;
  for(int n=0;n<N;n++){ const size_t idx=(size_t)(tokens[tok_start+n]+offsets[n]); acc+=vload_half(idx*dim+d, table); }
  out[d]=acc*scale;
}
