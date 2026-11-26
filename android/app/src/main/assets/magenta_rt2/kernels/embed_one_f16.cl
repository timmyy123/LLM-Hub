// Embed a single token id from tokbuf[q]: out[d] = table[tokbuf[q]][d] * sqrt(dim). Stays on GPU.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void embed_one_f16(__global const half* table, __global const int* tokbuf, const int q,
                            __global float* out, const int dim, const int vocab){
  const int d=get_global_id(0); if(d>=dim) return;
  int id=tokbuf[q]; if(id<0)id=0; if(id>=vocab)id=vocab-1;
  out[d]=vload_half((size_t)id*dim+d, table)*sqrt((float)dim);
}
