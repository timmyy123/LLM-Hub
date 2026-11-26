// split channels into a batch dim: out[g,t,f,c] = in[t,f, g*Cg+c]. in [T,F,G*Cg], out [G,T,F,Cg].
__kernel void split_ch_to_batch(__global const float* in, __global float* out,
                                const int T, const int F, const int G, const int Cg){
  const int c=get_global_id(0), f=get_global_id(1), gt=get_global_id(2);
  if(c>=Cg||f>=F||gt>=G*T) return;
  const int g=gt/T, t=gt%T, Ctot=G*Cg;
  out[(((size_t)g*T+t)*F+f)*Cg+c] = in[((size_t)t*F+f)*Ctot + g*Cg+c];
}
// concat batch back to channels: out[t,f, g*Cg+c] = in[g,t,f,c]. in [G,T,F,Cg], out [T,F,G*Cg].
__kernel void concat_batch_to_ch(__global const float* in, __global float* out,
                                 const int T, const int F, const int G, const int Cg){
  const int c=get_global_id(0), f=get_global_id(1), t=get_global_id(2);
  if(c>=Cg||f>=F||t>=T) return;
  for(int g=0;g<G;g++) out[((size_t)t*F+f)*(G*Cg)+ g*Cg+c] = in[(((size_t)g*T+t)*F+f)*Cg+c];
}
