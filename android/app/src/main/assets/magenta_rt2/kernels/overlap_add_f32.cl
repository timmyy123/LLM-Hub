// Windowed overlap-add: out[pos,ch] = sum_t frames[t, pos-t*hop, ch] for t with 0<=pos-t*hop<Nfft.
// frames [T,Nfft,2], out [T*hop,2]. (hop<Nfft ⇒ up to 2 frames overlap.)
//
// A chunk's first Nfft-hop samples receive only one window instead of two, so without the tail
// arguments every chunk ramps in from silence over 10 ms and the last frame's second half past
// T*hop is dropped. tail_in carries the previous chunk's dropped half into this one; tail_out saves
// this chunk's for the next.
//
// This was tried once with a STATELESS codec underneath and it banged: the two half-windows came
// from independent decodes (each chunk zero-padded on its left), so summing them at full amplitude
// was worse than the notch. It only reconstructs the waveform once every conv in the cascade
// carries its own context. That is now true, so this is on.
//
// tail_in and tail_out MUST be different buffers — separate parameters, one const, so the compiler
// may assume no aliasing and hoist the store above the load, at which point a chunk adds its own
// fresh tail to its own opening samples. The caller ping-pongs two buffers.
__kernel void overlap_add_f32(__global const float* frames, __global float* out,
                              const int T, const int Nfft, const int hop,
                              __global const float* tail_in, __global float* tail_out,
                              const int use_tail){
  const int pos=get_global_id(0), ch=get_global_id(1);
  const int Out=T*hop; if(pos>=Out||ch>=2) return;
  float acc=0.0f;
  for(int t=0;t<T;t++){ const int n=pos-t*hop; if(n>=0&&n<Nfft) acc+=frames[((size_t)t*Nfft+n)*2+ch]; }
  const int lap = Nfft - hop;
  if(pos < lap){
    if(use_tail) acc += tail_in[(size_t)pos*2+ch];
    tail_out[(size_t)pos*2+ch] = frames[((size_t)(T-1)*Nfft + hop + pos)*2+ch];
  }
  out[(size_t)pos*2+ch]=acc;
}
