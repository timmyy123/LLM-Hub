// Fake int8 quantization — simulate int8 (dot8) numerical precision while staying in fp32, so we
// can measure whether the phase-sensitive SpectroStream codec survives int8 BEFORE building the real
// qcom_dot8_acc pipeline. Per-row symmetric: buf laid out [ngroups, gsize]; one work-group per group.
//   scale = max|x| / 127 ;  x := clamp(round(x/scale), -127, 127) * scale
// For weights pass a row = one output channel (per-channel quant); for activations a row = one
// spatial position's K taps (per-row / per-token quant). gws = ngroups*WG, lws = WG (=64).
__kernel void fakeq8_rows(__global float* buf, const int gsize){
  const int g=get_group_id(0), lid=get_local_id(0), W=get_local_size(0);
  __local float sm[64];
  const size_t base=(size_t)g*gsize;
  float m=0.0f;
  for(int i=lid;i<gsize;i+=W) m=fmax(m,fabs(buf[base+i]));
  sm[lid]=m; barrier(CLK_LOCAL_MEM_FENCE);
  for(int s=W/2;s>0;s>>=1){ if(lid<s) sm[lid]=fmax(sm[lid],sm[lid+s]); barrier(CLK_LOCAL_MEM_FENCE); }
  float scale=sm[0]/127.0f; if(scale<=0.0f) scale=1.0f;
  const float inv=1.0f/scale;
  for(int i=lid;i<gsize;i+=W){ float q=round(buf[base+i]*inv); q=clamp(q,-127.0f,127.0f); buf[base+i]=q*scale; }
}
