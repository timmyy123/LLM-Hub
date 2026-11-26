// LayerNorm: out = (x-mean)/sqrt(var+eps)*weight + bias. One work-item per row.
// x [rows,dim] fp32 ; weight,bias [dim] fp16 ; out [rows,dim] fp32.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void layer_norm_f32(__global const float* x, __global const half* w, __global const half* b,
                             __global float* out, const int dim, const float eps){
  const int row=get_global_id(0); const size_t base=(size_t)row*dim;
  float mean=0.0f; for(int d=0;d<dim;d++) mean+=x[base+d]; mean/=dim;
  float var=0.0f; for(int d=0;d<dim;d++){float t=x[base+d]-mean; var+=t*t;} var/=dim;
  const float inv=rsqrt(var+eps);
  for(int d=0;d<dim;d++) out[base+d]=(x[base+d]-mean)*inv*vload_half(d,w)+vload_half(d,b);
}
