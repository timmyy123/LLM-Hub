// Reference: model_info/transformers_src/modeling_magenta_realtime_2.py:1-1 Placeholder (MLX conv1d semantics)
//
// conv_1d — dilation-aware 1D convolution, NCL layout (batch=1 implicit).
//
// Inputs:
//   in       — [C_in,  L_in]    storage_t
//   weight   — [C_out, C_in, K] storage_t   (groups=1)
//   bias     — [C_out]          storage_t   (optional)
// Output:
//   out      — [C_out, L_out]   storage_t
//
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

__kernel void conv_1d(
    __global const storage_t* in,
    __global const storage_t* weight,
    __global const storage_t* bias,
    __global       storage_t* out,
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
    for (int ic = 0; ic < C_in; ++ic) {
        const int w_base = oc * (C_in * K) + ic * K;
        const int x_base = ic * L_in;
        for (int k = 0; k < K; ++k) {
            const int il = ol * stride + k * dilation - padding;
            if ((uint)il >= (uint)L_in) continue;
            const float xv = LOAD(in, x_base + il);
            const float wv = LOAD(weight, w_base + k);
            acc += xv * wv;
        }
    }
    if (has_bias) acc += LOAD(bias, oc);
    STORE(out, gid, acc);
}
