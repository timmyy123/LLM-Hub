// Prefill out[m,n] = bias[n] (so CLBlastSgemm with beta=1 yields A@B.T + bias). flat M*N.
__kernel void broadcast_bias_f32(__global const float* bias, __global float* out, const int M, const int N){
  const int i=get_global_id(0); if(i>=M*N) return; out[i]=bias[i%N];
}
