// Elementwise residual add: out[i] = a[i] + b[i]  (fp32).
__kernel void add_f32(__global const float* a, __global const float* b, __global float* out, const int n){
  const int i=get_global_id(0); if(i>=n) return; out[i]=a[i]+b[i];
}
