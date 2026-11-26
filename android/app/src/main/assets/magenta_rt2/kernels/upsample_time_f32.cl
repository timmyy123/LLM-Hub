// Nearest-neighbor time upsample by rate r: out[to,f,c] = in[to/r, f, c]. in [T,F,C], out [T*r,F,C].
__kernel void upsample_time_f32(__global const float* in, __global float* out,
                                const int T, const int F, const int C, const int r){
  const int c=get_global_id(0), f=get_global_id(1), to=get_global_id(2);
  if(c>=C||f>=F||to>=T*r) return;
  out[((size_t)to*F+f)*C+c]=in[(((size_t)(to/r))*F+f)*C+c];
}
