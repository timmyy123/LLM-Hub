// Conv2D, fp32 weights, LDS weight-staging. Each WG = 1 Cout-tile (4 ch) × 1 time × FB freq.
// The 4 Cout weight rows for each (kh,kw) are loaded to LDS ONCE per WG and reused across FB freq
// outputs (weights are the bandwidth bottleneck). Input rides L1 (adjacent fo overlap). float4 inner.
// gws=(ceil(Cout/4), ceil(Fout/FB)*FB, Tout), lws=(1,FB,1). ws = local float[4*Cin].
__kernel void conv2d_lds_f32w(__global const float* in, __global const float* W, __global const float* bias,
    __global float* out, const int T, const int F, const int Cin, const int Tout, const int Fout,
    const int Cout, const int kH, const int kW, const int sh, const int sw, const int padT0, const int padF0,
    __local float* ws){
  const int co0=get_group_id(0)*4, to=get_group_id(2);
  const int lid=get_local_id(1), FB=get_local_size(1);
  const int fo=get_group_id(1)*FB+lid;
  const int c1=co0+1<Cout, c2=co0+2<Cout, c3=co0+3<Cout;
  float a0=bias[co0], a1=c1?bias[co0+1]:0.0f, a2=c2?bias[co0+2]:0.0f, a3=c3?bias[co0+3]:0.0f;
  for(int kh=0;kh<kH;kh++){ const int ti=to*sh+kh-padT0;
    for(int kw=0;kw<kW;kw++){
      // cooperatively stage 4 Cout weight rows for (kh,kw) into ws[c*Cin+ci]
      const size_t wkk=(((size_t)kh)*kW+kw)*Cin;
      for(int i=lid;i<4*Cin;i+=FB){ const int c=i/Cin, ci=i-c*Cin;
        ws[i]=(co0+c<Cout)? W[((size_t)(co0+c)*kH*kW)*Cin + wkk + ci] : 0.0f; }
      barrier(CLK_LOCAL_MEM_FENCE);
      if(ti>=0 && ti<T && fo<Fout){ const int fi=fo*sw+kw-padF0;
        if(fi>=0 && fi<F){ const size_t ib=((size_t)ti*F+fi)*Cin;
          for(int ci=0;ci<Cin;ci+=4){ float4 iv=vload4(0,in+ib+ci);
            a0+=dot(iv,vload4(0,ws+ci)); a1+=dot(iv,vload4(0,ws+Cin+ci));
            a2+=dot(iv,vload4(0,ws+2*Cin+ci)); a3+=dot(iv,vload4(0,ws+3*Cin+ci)); } } }
      barrier(CLK_LOCAL_MEM_FENCE);
    } }
  if(fo<Fout){ const size_t ob=((size_t)to*Fout+fo)*Cout+co0;
    out[ob]=a0; if(c1)out[ob+1]=a1; if(c2)out[ob+2]=a2; if(c3)out[ob+3]=a3; }
}
