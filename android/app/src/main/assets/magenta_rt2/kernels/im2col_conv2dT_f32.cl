// Batched im2col for Conv2DTranspose. in [B,T,F,Cin], col [B*Tout*Fout, K]. Row m → batch b.
// apply_elu (OPT #7): when 1, apply ELU (x>0?x:exp(x)-1) to each gathered value so the
// activation fuses into the gather — drops a standalone elu kernel + its full-size buffer
// before the convT. ELU(0)=0 so zero-padding is unchanged; the residual shortcut reads raw `in`.
__kernel void im2col_conv2dT_f32(__global const float* in, __global float* col, const int B,
    const int T, const int F, const int Cin, const int Tout, const int Fout,
    const int kH, const int kW, const int sh, const int sw, const int padT, const int padF,
    const int apply_elu){
  const int k=get_global_id(0), m=get_global_id(1);
  const int K=kH*kW*Cin, MB=B*Tout*Fout;
  if(k>=K || m>=MB) return;
  const int b=m/(Tout*Fout), mm=m-b*Tout*Fout, to=mm/Fout, fo=mm%Fout;
  const int ci=k%Cin, kk=k/Cin, kw=kk%kW, kh=kk/kW;
  const int nt=to+padT-kh, nf=fo+padF-kw;
  float v=0.0f;
  if(nt%sh==0 && nf%sw==0){ const int ti=nt/sh, fi=nf/sw;
    if(ti>=0&&ti<T&&fi>=0&&fi<F) v=in[(((size_t)b*T+ti)*F+fi)*Cin+ci]; }
  if(apply_elu) v = v>0.0f ? v : (exp(v)-1.0f);
  col[(size_t)m*K+k]=v;
}
