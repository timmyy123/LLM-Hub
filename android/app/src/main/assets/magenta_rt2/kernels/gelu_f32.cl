// Exact (erf) GELU: out = 0.5*x*(1+erf(x/sqrt(2))). One work-item per element.
__kernel void gelu_f32(__global const float* x, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i >= n) return;
  const float v = x[i];
  out[i] = 0.5f*v*(1.0f + erf(v*0.70710678118654752440f));
}
