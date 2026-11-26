// Conv2DTranspose, fp32 weights, 4 out-ch × 4 freq register tile (explicit scalars). float4 over Cin.
// Batched (OPT: fused/direct conv, no im2col col-buffer): gws.z = B*Tout, b=z/Tout, to=z%Tout.
// gws={ceil(Cout/4), ceil(Fout/4), B*Tout}.
#define DOTACC(a, iv, wv) a += dot(iv, wv)
#define ELU4(iv) iv = select(exp(iv)-(float4)1.0f, iv, iv>(float4)0.0f)
__kernel void conv2d_transpose_f32w(__global const float* in, __global const float* W, __global const float* bias,
    __global float* out, const int T, const int F, const int Cin, const int Tout, const int Fout,
    const int Cout, const int kH, const int kW, const int sh, const int sw, const int padT, const int padF,
    const int B, const int apply_elu){
  const int co0=get_global_id(0)*4, fo0=get_global_id(1)*4, z=get_global_id(2);
  const int b=z/Tout, to=z-b*Tout;
  if(co0>=Cout||fo0>=Fout||b>=B) return;
  const size_t in_bb=(size_t)b*T*F*Cin, out_bb=(size_t)b*Tout*Fout*Cout;
  const int c1=co0+1<Cout, c2=co0+2<Cout, c3=co0+3<Cout;
  float bb0=bias[co0], bb1=c1?bias[co0+1]:0.0f, bb2=c2?bias[co0+2]:0.0f, bb3=c3?bias[co0+3]:0.0f;
  float a00=bb0,a01=bb0,a02=bb0,a03=bb0, a10=bb1,a11=bb1,a12=bb1,a13=bb1,
        a20=bb2,a21=bb2,a22=bb2,a23=bb2, a30=bb3,a31=bb3,a32=bb3,a33=bb3;
  for(int kh=0;kh<kH;kh++){ const int num=to+padT-kh; if(num%sh!=0) continue; const int ti=num/sh; if(ti<0||ti>=T) continue;
    for(int kw=0;kw<kW;kw++){
      int n0=(fo0+0)+padF-kw, n1=(fo0+1)+padF-kw, n2=(fo0+2)+padF-kw, n3=(fo0+3)+padF-kw;
      int g0=n0/sw,g1=n1/sw,g2=n2/sw,g3=n3/sw;
      int v0=(n0%sw==0&&g0>=0&&g0<F), v1=(n1%sw==0&&g1>=0&&g1<F), v2=(n2%sw==0&&g2>=0&&g2<F), v3=(n3%sw==0&&g3>=0&&g3<F);
      const size_t wb0=(((size_t)(co0+0)*kH+kh)*kW+kw)*Cin, wb1=(((size_t)(co0+1)*kH+kh)*kW+kw)*Cin,
                   wb2=(((size_t)(co0+2)*kH+kh)*kW+kw)*Cin, wb3=(((size_t)(co0+3)*kH+kh)*kW+kw)*Cin;
      const size_t ib=in_bb+(size_t)ti*F*Cin;
      for(int ci=0;ci<Cin;ci+=4){
        float4 w0=vload4(0,W+wb0+ci), w1=c1?vload4(0,W+wb1+ci):(float4)0.0f,
               w2=c2?vload4(0,W+wb2+ci):(float4)0.0f, w3=c3?vload4(0,W+wb3+ci):(float4)0.0f;
        if(v0){ float4 iv=vload4(0,in+ib+(size_t)g0*Cin+ci); if(apply_elu)ELU4(iv); DOTACC(a00,iv,w0);DOTACC(a10,iv,w1);DOTACC(a20,iv,w2);DOTACC(a30,iv,w3); }
        if(v1){ float4 iv=vload4(0,in+ib+(size_t)g1*Cin+ci); if(apply_elu)ELU4(iv); DOTACC(a01,iv,w0);DOTACC(a11,iv,w1);DOTACC(a21,iv,w2);DOTACC(a31,iv,w3); }
        if(v2){ float4 iv=vload4(0,in+ib+(size_t)g2*Cin+ci); if(apply_elu)ELU4(iv); DOTACC(a02,iv,w0);DOTACC(a12,iv,w1);DOTACC(a22,iv,w2);DOTACC(a32,iv,w3); }
        if(v3){ float4 iv=vload4(0,in+ib+(size_t)g3*Cin+ci); if(apply_elu)ELU4(iv); DOTACC(a03,iv,w0);DOTACC(a13,iv,w1);DOTACC(a23,iv,w2);DOTACC(a33,iv,w3); }
      }
    } }
  const size_t ob=out_bb+(size_t)to*Fout*Cout;
  #define WR(cc,ff,val) if(co0+cc<Cout && fo0+ff<Fout) out[ob+(size_t)(fo0+ff)*Cout+co0+cc]=val;
  WR(0,0,a00)WR(0,1,a01)WR(0,2,a02)WR(0,3,a03) WR(1,0,a10)WR(1,1,a11)WR(1,2,a12)WR(1,3,a13)
  WR(2,0,a20)WR(2,1,a21)WR(2,2,a22)WR(2,3,a23) WR(3,0,a30)WR(3,1,a31)WR(3,2,a32)WR(3,3,a33)
}
