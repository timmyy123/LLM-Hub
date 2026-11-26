// RVQ codes → embeddings: out[t,d] = sum_{q<NCB} table[q][codes[t,q]][d].
// table [num_quant,codebook,dim] fp16 ; codes [T,NCB] int32 ; out [T,dim] fp32.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void rvq_embed_f16(__global const half* table, __global const int* codes,
                            __global float* out, const int T, const int NCB, const int CB, const int dim){
  const int t=get_global_id(0), d=get_global_id(1);
  if(t>=T||d>=dim) return;
  float acc=0.0f;
  for(int q=0;q<NCB;q++){ const int c=codes[t*NCB+q]; acc+=vload_half(((size_t)q*CB+c)*dim+d, table); }
  out[(size_t)t*dim+d]=acc;
}
