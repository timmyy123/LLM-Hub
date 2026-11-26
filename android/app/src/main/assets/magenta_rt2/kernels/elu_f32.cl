// ELU: out = x>0 ? x : exp(x)-1. In-place capable. One work-item per element.
__kernel void elu_f32(__global const float* x, __global float* out, const int n){
  const int i=get_global_id(0); if(i>=n) return; const float v=x[i];
  out[i] = v>0.0f ? v : (exp(v)-1.0f);
}
