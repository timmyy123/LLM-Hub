// Phase-split Conv2DTranspose for the fp16 codec GEMM path.
//
// im2col_conv2dT_f16 builds the FULL [M, K] column matrix of a transposed conv, K = kH*kW*Cin. With
// stride (sh, sw) an output position only receives input from taps where kh ≡ (to+padT) (mod sh)
// and kw ≡ (fo+padF) (mod sw); every other entry of its row is zero. The codec's six transposed
// convs have strides (2,2), (1,2), (1,2), (1,3), (1,2), (1,2), so 50-75% of that matrix — and of the
// HGEMM that consumes it — is zeros multiplied by weights.
//
// Split the outputs by phase (rt, rf) = ((to+padT) mod sh, (fo+padF) mod sw). Inside one phase the
// set of live taps is fixed: kh = rt + i*sh, kw = rf + j*sw. So each phase is a DENSE GEMM with
// K' = nkh*nkw*Cin over the phase's own outputs, against the matching rows of the weight matrix.
// Same products, same sums (only the zero terms are gone), so the result is the same convolution.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// col[m', k'] for one phase. m' = (b, jt, jf) → to = to0 + jt*sh, fo = fo0 + jf*sw.
// k' = (ikh, ikw, ci) → kh = rt + ikh*sh, kw = rf + ikw*sw; ti = (to+padT-kh)/sh (exact by phase).
__kernel void im2col_convT_phase_f16(__global const float* in, __global half* col,
    const int B, const int T, const int F, const int Cin,
    const int Tp, const int Fp, const int to0, const int fo0,
    const int rt, const int rf, const int nkh, const int nkw,
    const int sh, const int sw, const int padT, const int padF,
    const int apply_elu, __global const float* sc){
  const int k = get_global_id(0), m = get_global_id(1);
  const int Kp = nkh*nkw*Cin, Mp = B*Tp*Fp;
  if (k >= Kp || m >= Mp) return;
  const int b = m/(Tp*Fp), mm = m - b*Tp*Fp, jt = mm/Fp, jf = mm - jt*Fp;
  const int to = to0 + jt*sh, fo = fo0 + jf*sw;
  const int ci = k % Cin, kk = k / Cin, ikw = kk % nkw, ikh = kk / nkw;
  const int kh = rt + ikh*sh, kw = rf + ikw*sw;
  const int ti = (to + padT - kh) / sh, fi = (fo + padF - kw) / sw;
  float v = 0.0f;
  if (ti >= 0 && ti < T && fi >= 0 && fi < F) v = in[(((size_t)b*T + ti)*F + fi)*Cin + ci];
  if (apply_elu) v = v > 0.0f ? v : (exp(v) - 1.0f);
  vstore_half(v * sc[1], (size_t)m*Kp + k, col);
}

// W'[k', n] = W[k(kh,kw,ci), n] for one phase — the live rows of the [K, N] fp16 weight matrix,
// k = (kh*kW + kw)*Cin + ci as im2col_conv2dT_f16 lays them out. Built once per weight and phase.
__kernel void convT_phase_weights_f16(__global const half* W, __global half* Wp,
    const int N, const int Cin, const int kW, const int rt, const int rf,
    const int nkw, const int sh, const int sw, const int Kp){
  const int k = get_global_id(0), n = get_global_id(1);
  if (k >= Kp || n >= N) return;
  const int ci = k % Cin, kk = k / Cin, ikw = kk % nkw, ikh = kk / nkw;
  const int kh = rt + ikh*sh, kw = rf + ikw*sw;
  const size_t src = ((size_t)(kh*kW + kw)*Cin + ci) * N + n;
  Wp[(size_t)k*N + n] = W[src];
}

// out[(b*Tout+to)*Fout+fo, n] = C'[m', n] * scale — the phase's rows back to their output positions.
__kernel void convT_phase_scatter_f16(__global const half* Cp, __global float* out,
    const int N, const int Tp, const int Fp, const int to0, const int fo0,
    const int sh, const int sw, const int Tout, const int Fout, const int Mp,
    __global const float* sc){
  const int n = get_global_id(0), m = get_global_id(1);
  if (n >= N || m >= Mp) return;
  const int b = m/(Tp*Fp), mm = m - b*Tp*Fp, jt = mm/Fp, jf = mm - jt*Fp;
  const int to = to0 + jt*sh, fo = fo0 + jf*sw;
  out[(((size_t)b*Tout + to)*Fout + fo)*N + n] = vload_half((size_t)m*N + n, Cp) * sc[0];
}
