// IRFFT(960) + synthesis window per frame, per stereo channel.
// dec [T,Nbins,Cdec] (Cdec=4 = re0,im0,re1,im1); padded Nyquist bin = 0 (not included).
// out frames [T,Nfft,2]: frames[t,n,ch] = window[n] * (1/Nfft) * (re[0] + 2*sum_{k=1}^{Nbins-1}(re[k]cos - im[k]sin))
//   (im[0]*sin(0)=0 so DC imag drops naturally; Nyquist padded to 0).
#define PI2 6.28318530717958647692f
__kernel void irfft_window_f32(__global const float* dec, __global const float* win,
    __global float* out, const int T, const int Nbins, const int Cdec, const int Nfft){
  const int n=get_global_id(0), ch=get_global_id(1), t=get_global_id(2);
  if(n>=Nfft||ch>=2||t>=T) return;
  const float base=PI2*(float)n/(float)Nfft;
  float acc=0.0f;
  for(int k=0;k<Nbins;k++){
    const size_t idx=((size_t)t*Nbins+k)*Cdec + 2*ch;
    const float re=dec[idx], im=dec[idx+1];
    const float w=(k==0)?1.0f:2.0f;
    const float ang=base*(float)k;
    acc += w*(re*cos(ang) - im*sin(ang));
  }
  out[((size_t)t*Nfft+n)*2+ch] = (acc/(float)Nfft) * win[n];
}
