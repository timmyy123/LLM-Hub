// im2col gathers with 128-bit transactions.
//
// The scalar gather has each work item read ONE float and write ONE float. Adjacent items already
// read adjacent addresses, so the access is coalesced — but every transaction is 32-bit, a quarter
// of the 128-bit maximum the memory system can move in one go (80-NB295-11 §6.3). Routing the same
// reads through the texture L1 was measured and does nothing for exactly this reason: the problem is
// transaction WIDTH, not cache residency.
//
// Here each work item handles four consecutive k. Because ci = k % Cin and every codec Cin is a
// multiple of 4, those four share kh/kw and differ only in ci — so the source is four consecutive
// floats (a float4 load, 16B-aligned since Cin % 4 == 0) and the destination is four consecutive
// floats in the same row (a float4 store). Work items drop 4x and every transaction is full width.
__kernel void im2col_conv2d_f32_v4(__global const float* in, __global float* col, const int B,
    const int T, const int F, const int Cin, const int Tout, const int Fout,
    const int kH, const int kW, const int sh, const int sw, const int padT0, const int padF0,
    const int apply_elu){
  const int k4=get_global_id(0), m=get_global_id(1);
  const int K=kH*kW*Cin, MB=B*Tout*Fout;
  const int k=k4<<2;
  if(k>=K || m>=MB) return;
  const int b=m/(Tout*Fout), mm=m-b*Tout*Fout, to=mm/Fout, fo=mm%Fout;
  const int ci=k%Cin, kk=k/Cin, kw=kk%kW, kh=kk/kW;
  const int ti=to*sh+kh-padT0, fi=fo*sw+kw-padF0;
  float4 v=(float4)(0.0f);
  if(ti>=0 && ti<T && fi>=0 && fi<F)
    v=vload4(0, in+(((size_t)b*T+ti)*F+fi)*Cin+ci);
  if(apply_elu){
    v.x = v.x>0.0f ? v.x : (exp(v.x)-1.0f);
    v.y = v.y>0.0f ? v.y : (exp(v.y)-1.0f);
    v.z = v.z>0.0f ? v.z : (exp(v.z)-1.0f);
    v.w = v.w>0.0f ? v.w : (exp(v.w)-1.0f);
  }
  vstore4(v, 0, col+(size_t)m*K+k);
}

// Transposed variant. The four lanes share kh/kw, so they also share the stride test — one branch
// for the group, not four.
__kernel void im2col_conv2dT_f32_v4(__global const float* in, __global float* col, const int B,
    const int T, const int F, const int Cin, const int Tout, const int Fout,
    const int kH, const int kW, const int sh, const int sw, const int padT, const int padF,
    const int apply_elu){
  const int k4=get_global_id(0), m=get_global_id(1);
  const int K=kH*kW*Cin, MB=B*Tout*Fout;
  const int k=k4<<2;
  if(k>=K || m>=MB) return;
  const int b=m/(Tout*Fout), mm=m-b*Tout*Fout, to=mm/Fout, fo=mm%Fout;
  const int ci=k%Cin, kk=k/Cin, kw=kk%kW, kh=kk/kW;
  const int nt=to+padT-kh, nf=fo+padF-kw;
  float4 v=(float4)(0.0f);
  if(nt%sh==0 && nf%sw==0){ const int ti=nt/sh, fi=nf/sw;
    if(ti>=0&&ti<T&&fi>=0&&fi<F)
      v=vload4(0, in+(((size_t)b*T+ti)*F+fi)*Cin+ci); }
  if(apply_elu){
    v.x = v.x>0.0f ? v.x : (exp(v.x)-1.0f);
    v.y = v.y>0.0f ? v.y : (exp(v.y)-1.0f);
    v.z = v.z>0.0f ? v.z : (exp(v.z)-1.0f);
    v.w = v.w>0.0f ? v.w : (exp(v.w)-1.0f);
  }
  vstore4(v, 0, col+(size_t)m*K+k);
}
