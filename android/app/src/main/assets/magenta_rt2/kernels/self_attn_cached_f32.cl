// Cached self-attention: query at current position attends to [optional sink, cached 0..pos].
// qkv [3*H*D] current (q,k,v); kcache/vcache [maxpos,H,D] (current k,v already written at pos).
// sink_k/sink_v/per_dim_scale fp16. out [H*D] fp32. One work-item per head.
// scale_vec[d] = r_softplus_0 * (1/sqrt(D)) * softplus(per_dim_scale[d]); sink uses unscaled q.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void self_attn_cached_f32(
    __global const float* qkv, __global const float* kcache, __global const float* vcache,
    __global const half* sink_k, __global const half* sink_v, __global const half* pds,
    __global float* out, const int H, const int D, const int pos, const int has_sink,
    const float inv_sqrt_d, const int max_past){
  const int h=get_global_id(0); if(h>=H) return;
  const int HD=H*D; const size_t qb=(size_t)h*D;
  const float r=1.442695041f;
  // local attention window: attend to [lo..pos] (+ optional sink). max_past = horizon.
  int lo = pos - max_past; if(lo<0) lo=0;
  float scores[128];
  const int ns = has_sink?1:0;
  const int n  = ns + (pos-lo+1);
  float maxs=-1e30f;
  if(has_sink){ float s=0.0f; for(int d=0;d<D;d++) s+=qkv[qb+d]*vload_half(qb+d,sink_k); scores[0]=s; if(s>maxs)maxs=s; }
  for(int p=lo;p<=pos;p++){
    float s=0.0f;
    for(int d=0;d<D;d++){ const float sc=r*inv_sqrt_d*log1p(exp(vload_half(d,pds))); s+=(qkv[qb+d]*sc)*kcache[(size_t)p*HD+qb+d]; }
    scores[ns+(p-lo)]=s; if(s>maxs)maxs=s;
  }
  float Z=0.0f; for(int i=0;i<n;i++){ scores[i]=exp(scores[i]-maxs); Z+=scores[i]; }
  for(int d=0;d<D;d++){
    float acc=0.0f;
    if(has_sink) acc+=scores[0]*vload_half(qb+d,sink_v);
    for(int p=lo;p<=pos;p++) acc+=scores[ns+(p-lo)]*vcache[(size_t)p*HD+qb+d];
    out[qb+d]=acc/Z;
  }
}
