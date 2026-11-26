// Reference: model_info/transformers_src/modeling_magenta_realtime_2.py:1-1 Placeholder (MLX conv_transpose1d semantics)
//
// conv_transpose_1d — transposed (fractionally-strided) 1D convolution.
//
// Semantics: matches standard ConvTranspose1d formula (output_padding=0, groups=1).
//
//   L_out = (L_in - 1) * stride - 2*padding + dilation*(K - 1) + 1
//
// Implementation: each output position 'ol' "gathers" the input positions
// 'il' whose forward-conv kernel taps would have landed on ol. Concretely:
//
//   il*stride - padding + k*dilation == ol      (forward sense)
// ⇒ il = (ol + padding - k*dilation) / stride   (integer; must be exact)
//
// We loop over k, check the integer-divisibility condition, accumulate.
// Layout matches conv_1d for consistency.
//
// One work-item per output element. C_out * L_out total.

// DTYPE TEMPLATE (NNOPT fp16/fp32): use storage_t + LOAD/STORE.

#ifdef USE_FP16
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  typedef half storage_t;
  #define LOAD(p,i)   vload_half((i),(p))
  #define STORE(p,i,v) vstore_half((v),(i),(p))
#else
  typedef float storage_t;
  #define LOAD(p,i)   ((p)[(i)])
  #define STORE(p,i,v) ((p)[(i)] = (v))
#endif

__kernel void conv_transpose_1d(
    __global const storage_t* in,           // [C_in,  L_in]
    __global const storage_t* weight,       // [C_in,  C_out, K]
    __global const storage_t* bias,         // [C_out] or NULL
    __global       storage_t* out,          // [C_out, L_out]
    const int C_in,
    const int C_out,
    const int L_in,
    const int L_out,
    const int K,
    const int stride,
    const int padding,
    const int dilation,
    const int has_bias) {

    const int gid = (int)get_global_id(0);
    const int total = C_out * L_out;
    if (gid >= total) return;

    const int oc = gid / L_out;
    const int ol = gid - oc * L_out;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int num = ol + padding - k * dilation;
        if (num < 0) continue;
        if ((num % stride) != 0) continue;
        const int il = num / stride;
        if ((uint)il >= (uint)L_in) continue;
        for (int ic = 0; ic < C_in; ++ic) {
            const int x_idx = ic * L_in + il;
            const int w_idx = ic * (C_out * K) + oc * K + k;
            const float xv = LOAD(in, x_idx);
            const float wv = LOAD(weight, w_idx);
            acc += xv * wv;
        }
    }
    if (has_bias) acc += LOAD(bias, oc);
    STORE(out, gid, acc);
}
