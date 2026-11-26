// Conv2DTranspose (channels-last, gather form). in [T,F,Cin], W [Cout,kH,kW,Cin], bias [Cout].
// out [Tout,Fout,Cout] (Tout=T*sh, Fout=F*sw for 'same'). padT/padF per layer.
//   ti = (to+padT-kh)/sh  (kept when divisible by sh and in [0,T)); fi likewise.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void conv2d_transpose_f16(__global const float* in, __global const half* W, __global const half* bias,
    __global float* out, const int T, const int F, const int Cin, const int Tout, const int Fout,
    const int Cout, const int kH, const int kW, const int sh, const int sw, const int padT, const int padF){
  const int co=get_global_id(0), fo=get_global_id(1), to=get_global_id(2);
  if(co>=Cout||fo>=Fout||to>=Tout) return;
  float acc=vload_half(co,bias);
  for(int kh=0;kh<kH;kh++){
    const int num=to+padT-kh; if(num%sh!=0) continue; const int ti=num/sh; if(ti<0||ti>=T) continue;
    for(int kw=0;kw<kW;kw++){
      const int numf=fo+padF-kw; if(numf%sw!=0) continue; const int fi=numf/sw; if(fi<0||fi>=F) continue;
      const size_t inb=((size_t)ti*F+fi)*Cin, wb=(((size_t)co*kH+kh)*kW+kw)*Cin;
      for(int ci=0;ci<Cin;ci++) acc+=in[inb+ci]*vload_half(wb+ci,W);
    }
  }
  out[((size_t)to*Fout+fo)*Cout+co]=acc;
}
