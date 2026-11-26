// Batched nearest 2D upsample by (rt,rf). in [B,T,F,C], out [B,T*rt,F*rf,C].
__kernel void upsample2d_f32(__global const float* in, __global float* out,
                             const int B, const int T, const int F, const int C, const int rt, const int rf){
  const int c=get_global_id(0), fo=get_global_id(1), bt=get_global_id(2);  // bt = b*(T*rt)+to
  const int Tu=T*rt, Fu=F*rf;
  if(c>=C||fo>=Fu||bt>=B*Tu) return;
  const int b=bt/Tu, to=bt%Tu;
  out[(((size_t)b*Tu+to)*Fu+fo)*C+c]=in[(((size_t)b*T+(to/rt))*F+(fo/rf))*C+c];
}
