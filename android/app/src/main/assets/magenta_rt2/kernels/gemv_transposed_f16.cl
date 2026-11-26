// Transposed-weight coalesced GEMV: out[n] = sum_k x[k]*Wt[k,n]. Wt is [K,N] (transposed).
// One thread per output n → adjacent threads read Wt[k*N + n..] = coalesced wavefront read. NO reduction.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemv_transposed_f16(__global const float* x, __global const half* Wt,
                                  __global float* out, const int K, const int N){
  const int n=get_global_id(0); if(n>=N) return;
  float acc=0.0f;
  for(int k=0;k<K;k++) acc += x[k]*vload_half((size_t)k*N+n, Wt);
  out[n]=acc;
}

// Same K-major idea, but each thread owns FOUR consecutive outputs and reads them as one half4.
// The scalar version above is latency-bound: 2 bytes per thread per k, on a serial accumulate chain
// with nothing else in flight. Four outputs give four independent accumulators (the Adreno ceiling —
// more than four spills) and widen each thread's read to 8 bytes, so a 64-wide wave pulls 512
// contiguous bytes per step instead of 128.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void gemv_transposed_f16_v4(__global const float* x, __global const half* Wt,
                                     __global float* out, const int K, const int N){
  const int n4 = get_global_id(0);
  const int n = n4 << 2;
  if (n >= N) return;
  float4 acc = (float4)(0.0f);
  for (int k = 0; k < K; k++) {
    const float xv = x[k];
    const float4 w = vload_half4(0, Wt + (size_t)k * N + n);
    acc += xv * w;
  }
  if (n + 3 < N) { vstore4(acc, 0, out + n); }
  else { for (int u = 0; u + n < N; ++u) out[n+u] = ((const float*)&acc)[u]; }
}
