// Conv2D (channels-last): in [T,F,Cin] fp32, W [Cout,kH,kW,Cin] fp16, bias [Cout] fp16,
// out [Tout,Fout,Cout] fp32. Correlation. padT0/padF0 = left/top pad (semicausal time, same freq).
// out[to,fo,co] = bias[co] + sum_{kh,kw,ci} in[to*sh+kh-padT0, fo*sw+kw-padF0, ci] * W[co,kh,kw,ci]
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void conv2d_f16(__global const float* in, __global const half* W, __global const half* bias,
    __global float* out, const int T, const int F, const int Cin, const int Tout, const int Fout,
    const int Cout, const int kH, const int kW, const int sh, const int sw, const int padT0, const int padF0){
  const int co=get_global_id(0), fo=get_global_id(1), to=get_global_id(2);
  if(co>=Cout||fo>=Fout||to>=Tout) return;
  float acc=vload_half(co,bias);
  for(int kh=0;kh<kH;kh++){
    const int ti=to*sh+kh-padT0; if(ti<0||ti>=T) continue;
    for(int kw=0;kw<kW;kw++){
      const int fi=fo*sw+kw-padF0; if(fi<0||fi>=F) continue;
      const size_t inb=((size_t)ti*F+fi)*Cin;
      const size_t wb=(((size_t)co*kH+kh)*kW+kw)*Cin;
      for(int ci=0;ci<Cin;ci++) acc+=in[inb+ci]*vload_half(wb+ci,W);
    }
  }
  out[((size_t)to*Fout+fo)*Cout+co]=acc;
}
