// Batched im2col for Conv2D writing fp16, scaled. Same gather as im2col_conv2d_f32.cl.
//
// The codec cascade is GEMM-bound (~154 GFLOP/s against the Adreno 840's ~3.7 TFLOPS fp32), so the
// lever is fp16: half the operand traffic and 2x the ALU rate. The obstacle was range, not
// precision — the cascade's late blocks reach |x| ~ 9.5e6 and fp16 stops at 65504. Convolution is
// linear, so dividing the input by `inv_scale` here and multiplying the result back afterwards is
// exact when the scale is a power of two, and it moves the whole block into fp16's range.
//
// ELU still fuses into the gather (ELU(0)=0, so zero-padding is unchanged), and it is applied
// BEFORE scaling because ELU is not linear.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void im2col_conv2d_f16(__global const float* in, __global half* col, const int B,
    const int T, const int F, const int Cin, const int Tout, const int Fout,
    const int kH, const int kW, const int sh, const int sw, const int padT0, const int padF0,
    const int apply_elu, __global const float* sc){
  const int k=get_global_id(0), m=get_global_id(1);
  const int K=kH*kW*Cin, MB=B*Tout*Fout;
  if(k>=K || m>=MB) return;
  const int b=m/(Tout*Fout), mm=m-b*Tout*Fout, to=mm/Fout, fo=mm%Fout;
  const int ci=k%Cin, kk=k/Cin, kw=kk%kW, kh=kk/kW;
  const int ti=to*sh+kh-padT0, fi=fo*sw+kw-padF0;
  float v = (ti>=0&&ti<T&&fi>=0&&fi<F) ? in[(((size_t)b*T+ti)*F+fi)*Cin+ci] : 0.0f;
  if(apply_elu) v = v>0.0f ? v : (exp(v)-1.0f);
  vstore_half(v * sc[1], (size_t)m*K+k, col);
}

// Batched im2col for Conv2D**Transpose**, writing fp16, scaled. Mirrors im2col_conv2dT_f32.cl.
//
// THIS KERNEL'S ABSENCE IS WHY THE fp16 CODEC RENDERED SILENCE. conv_clblast takes its gather from
// the caller precisely because a transposed convolution needs different index arithmetic — plain
// walks FORWARD (ti = to*sh + kh - pad, always reads), transposed walks BACKWARD and only reads
// where the stride divides evenly. The fp16 path had one hardcoded gather, the plain one, and used
// it for every conv including the conv-transpose that upsamples inside all six cascade blocks.
//
// The symptom is not a crash: it is wrong audio AND a fake speed-up, because the mis-computed
// indices fall out of range and the bounds test skips the work. The cascade's magnitudes then never
// climb (measured: scales topped out at 2^8 where a healthy run reaches 2^18+), so every later
// block decodes wreckage and the output is silent.
__kernel void im2col_conv2dT_f16(__global const float* in, __global half* col, const int B,
    const int T, const int F, const int Cin, const int Tout, const int Fout,
    const int kH, const int kW, const int sh, const int sw, const int padT, const int padF,
    const int apply_elu, __global const float* sc){
  const int k=get_global_id(0), m=get_global_id(1);
  const int K=kH*kW*Cin, MB=B*Tout*Fout;
  if(k>=K || m>=MB) return;
  const int b=m/(Tout*Fout), mm=m-b*Tout*Fout, to=mm/Fout, fo=mm%Fout;
  const int ci=k%Cin, kk=k/Cin, kw=kk%kW, kh=kk/kW;
  const int nt=to+padT-kh, nf=fo+padF-kw;
  float v=0.0f;
  if(nt%sh==0 && nf%sw==0){ const int ti=nt/sh, fi=nf/sw;
    if(ti>=0&&ti<T&&fi>=0&&fi<F) v=in[(((size_t)b*T+ti)*F+fi)*Cin+ci]; }
  // ELU before scaling: ELU is not linear, so elu(x/s) != elu(x)/s.
  if(apply_elu) v = v>0.0f ? v : (exp(v)-1.0f);
  vstore_half(v * sc[1], (size_t)m*K+k, col);
}
