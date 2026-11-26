// GEMV + bias with an optional RMS pre-norm folded in and an optional exact-GELU epilogue.
//
// The AR issues ~455 dispatches per frame, 45,500 per 2 s chunk, and roughly a quarter of them are
// tiny elementwise kernels bracketing the GEMVs: an RMSNorm over ~1024 floats before, a GELU over
// ~4096 after. Each pays a full dispatch to do microseconds of work. Measured on the 840: one gemv
// streams 6.29 MB in 0.115 ms (54.7 GB/s, 64% of peak), so at that rate the AR's 951 MB/frame should
// take 1.74 s/chunk — it takes 3.29 s. The missing ~1.5 s is these small dispatches, not the math.
//
// Pre-norm folds in EXACTLY because the normalisation constant is uniform over the row:
//     rms(x)[d] = x[d] * inv * g[d]           with inv = rsqrt(mean(x^2) + eps)
//     y[n] = sum_d W[n,d] * rms(x)[d] = inv * sum_d W[n,d] * (x[d] * g[d])
// so `inv` comes out of the dot product and is applied once at the end. Every workgroup already
// streams all of x for its own dot products, so sum(x^2) is free — it rides along in the same
// registers and reuses the reduction that is already there (one extra slot, no extra barrier).
//
// PRENORM=1 enables the fold; GELU=1 the epilogue; BIAS=1 the bias add. Compile-time so a kernel
// that uses none of them is byte-for-byte the original loop.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WG 64
// NOUT — outputs computed per WORKGROUP, and therefore the divisor on how many workgroups a GEMV
// dispatch launches: workgroups = out_dim / NOUT.
//
// This is a HOST-side lever, not a GPU one. The AR is limited by CPU time inside
// clEnqueueNDRangeKernel (12.11 ms of GPU work inside a 30.7 ms frame), and the driver appears to
// build its command stream per workgroup: on the 840, a 1-workgroup argmax enqueues in 17.6 us
// while a several-hundred-workgroup gemv takes 39-61 us for the identical API call. At NOUT=8 an
// out_dim=4096 GEMV launches 512 workgroups; NOUT=32 launches 128.
//
// The accumulators are NAMED, never an array — see the note below. NOUT=32 means 32 live floats per
// work item, which is real register pressure, so gpu_ms is on the Engine report card to catch a
// spill immediately rather than in a log nobody reads.
#ifndef NOUT
#define NOUT 8
#endif
#ifndef PRENORM
#define PRENORM 0
#endif
#ifndef GELU
#define GELU 0
#endif
// RESADD=1: fold the PREVIOUS block's post-norm + residual-add into this GEMV's input load, so
// `out = resid + rms2(proj)` costs no dispatch of its own. The AR is host-issue-bound, and the
// standalone rms_norm_add_wg_f32 is 84 dispatches/frame for 0.54 ms of actual GPU work — launch
// cost wearing a hat. Here `x` IS proj (un-normalised); the true block output is materialised once
// by workgroup 0 for the next block's residual, and every workgroup recomputes it into its own
// registers because each one streams the whole input vector anyway.
#ifndef RESADD
#define RESADD 0
#endif
#ifndef BIAS
#define BIAS 0
#endif
// INT8=1: weights are symmetric per-output-row int8 with an fp32 scale per row. The depth body's
// MLPs are read 12x per frame (once per RVQ level) and are the single largest slice of AR traffic;
// halving their bytes also drops the per-level working set under the 840's 18 MB HPM, so levels
// 1..11 can hit cache instead of DRAM. Compute is unchanged — this buys bandwidth, not FLOPs.
#ifndef INT8
#define INT8 0
#endif

// WIDE=1: stream the WEIGHT row in 128-bit transactions.
//
// This kernel is memory-bound on the weight operand — x is one row that every workgroup keeps in
// cache, W is a fresh megabyte per call. The load width therefore IS the kernel. The original loop
// read vload_half4 (64 bits) per lane per step, and the int8 variant read char4 — THIRTY-TWO bits,
// a quarter of what the bus can move in one transaction. That is why int8 measured the same
// per-call time as fp16 on the 620 despite reading half the bytes: it was not bandwidth-limited at
// all, it was transaction-limited. The Adreno guide names 128 bits as the maximum width and calls
// out memory-bound kernels specifically (80-NB295-11 Rev. C §6.3).
//
// So a step now covers 8 halves or 16 chars — 16 bytes either way. The tail loop below keeps the
// old 4-wide step for any in_dim the wide step does not divide, so this is safe for every shape
// rather than only the multiples-of-64 the AR happens to use today. WIDE=0 restores the old loop
// exactly (nvec becomes 0 and everything falls to the tail), which is the A/B.
#ifndef WIDE
#define WIDE 1
#endif

// DOT8 — DISABLED, BROKEN. The host never sets it (see use_dot8 in nnopt_gemv_fused, which is a
// hard-coded false with the full explanation). This path produced garbage audio on the Adreno 840;
// the prime suspect is the operand signedness of qcom_dot8_acc, which §9.5.1 documents one way and
// contradicts in its own example. Left in the tree because the algebra and the per-row correction
// are right and worth keeping; do not enable it without checking the output against fp16.
//
// DOT8=1: use qcom_dot8_acc — one instruction for four int8xint8 products plus a saturating int32
// accumulate (guide §9.5.1). It exists because §8 is explicit that "Adreno GPUs do not have general
// 8-bit ALU support", so the INT8 path's convert_float4 is converting int8 to float purely to reach
// a float dot() — ALU spent working around a missing instruction the chip actually has.
//
// The instruction is SIGNED x UNSIGNED. Weights are already signed int8; the activation therefore
// carries a zero point of 128 to become unsigned, and the resulting correction expands to
//     sum_d Wq[d]*(xq[d]-128) = sum_d Wq[d]*xq[d] - 128*sum_d Wq[d]
// whose second term is a per-row weight constant (`rowsum`), computed once at quantization time.
//
// Cost: the activation scale needs max|x| BEFORE the dot products, so this variant makes a first
// pass over x and one extra barrier. x is in_dim floats (3-16 KB) and every workgroup reads it
// anyway, so the pass is cheap; the barrier is not free (§6.1.4) and is why this is measured, not
// assumed.
#ifndef DOT8
#define DOT8 0
#endif

// IMG=1: fetch the int8 weights through an image2d VIEW of the same buffer rather than as raw
// global memory. Guide §6.2 — "Adreno GPUs have a powerful texture engine and dedicated level 1
// cache that can load data in image objects effectively" — and a buffer read never uses that L1.
// One CL_RGBA/CL_SIGNED_INT32 texel is 128 bits, the widest fetch §6.3 describes, and it holds
// exactly the sixteen weights a wide step consumes. Same bytes, same arithmetic; only the path
// they arrive through changes. The host leaves this off when the image could not be created.
#ifndef IMG
#define IMG 0
#endif

// CONSTX=1: the activation vector in __constant. Guide §6.4 — Adreno has fast on-chip constant
// memory, and the compiler can promote a __constant argument into it when max_constant_size tells
// it the size fits. x is at most 4096 floats (16 KB) and is re-read by every workgroup in the
// dispatch, which is the case this memory exists for.
//
// The host probes it rather than assuming: not every driver accepts a pooled CL_MEM_READ_WRITE
// buffer on a __constant argument, and the failure is at clSetKernelArg, not at build time.
#ifndef CONSTX
#define CONSTX 0
#endif
#if CONSTX
#define XSPACE __constant
#define XATTR  __attribute__((max_constant_size(16384)))
#else
#define XSPACE __global const
#define XATTR
#endif
#if DOT8
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable
#endif
#if INT8
#define WSTEP 16
#else
#define WSTEP 8
#endif
#define NV (WSTEP/4)     // float4 chunks of x consumed per wide step
//
// NOTE the wide body below uses NAMED vector variables, never `float4 xv[NV]`. The array version
// measured 28x SLOWER on the 620 (gemv_fused 407 us/call -> 11428 us/call): an indexed private
// array the compiler declines to promote lands in scratch memory, so every step round-tripped x
// through DRAM. Anything added here must stay indexable only by the unrolled NOUT loop.

#if IMG
const sampler_t kWSmp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
#endif

__kernel void linear_fused_f32(XSPACE float* x XATTR,
#if INT8
#if IMG
                               __read_only image2d_t Wimg,
#else
                               __global const char*  W,
#endif
                               __global const float* wscale,   // [out_dim], symmetric, absmax/127
#if DOT8
                               __global const int*   wrowsum,  // [out_dim], sum of the quantized row
#endif
#else
                               __global const half*  W,
#endif
#if BIAS
                               __global const half*  bias,
#endif
#if PRENORM
                               __global const half*  rms_scale,
                               const float           rms_eps,
#endif
#if RESADD
                               __global const float* resid,       // pre-block input to add
                               __global const half*  rms2_scale,  // post-norm weight
                               const float           rms2_eps,
                               __global float*       out_resid,   // materialised block output
#endif
                               __global float* out, const int in_dim, const int out_dim){
  // ls holds NOUT partial dot products, plus one more lane-set for sum(x^2) when pre-norming.
  __local float ls[WG*(NOUT+PRENORM)];
  const int row=get_global_id(0), tid=get_local_id(1), wg=get_group_id(1), n0=wg*NOUT;
  const size_t xb=(size_t)row*in_dim; const int in4=in_dim>>2;
  // NAMED accumulators, not `float acc[NOUT]`. An indexed private array is register-promotable
  // only while the compiler has registers to spare; the wide loop below raises live pressure past
  // that point and the array silently moves to scratch, which measured 18x slower on the 620 (407
  // -> 7531 us/call) — a memory round-trip per multiply-accumulate. Naming them removes the
  // decision from the compiler.
  // Repetition macros: NOUT named scalars, expanded at compile time. An indexed private array is
  // register-promotable only while the compiler has registers to spare, and it silently moves to
  // scratch memory when it does not — measured 18x slower on the 620 before this was named.
#define REP8(M)  M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#define REP16(M) REP8(M) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#define REP32(M) REP16(M) M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23) \
                          M(24) M(25) M(26) M(27) M(28) M(29) M(30) M(31)
#if   NOUT == 32
#define REP(M) REP32(M)
#elif NOUT == 16
#define REP(M) REP16(M)
#else
#define REP(M) REP8(M)
#endif
#define DECL_ACC(u) float a##u = 0.0f;
  REP(DECL_ACC)
#if PRENORM
  float ss=0.0f;
#endif
#if RESADD
  // BIT-EXACTNESS. rms_norm_add_wg_f32 sums with WG=128 SCALAR lanes striding by 128, then a
  // 128-wide tree. This kernel's workgroup is 64 wide and loads float4s, so the obvious
  // `ss += dot(xv,xv)` would sum in a different order and shift `inv` by an ulp — which flips an
  // argmax in an AR model (that kernel's own header says so). Each work item therefore carries TWO
  // virtual lanes, keeps their partials apart, and runs the identical tree. Same order, same bits.
  __local float rs2[128];
  {
    const size_t rb2=(size_t)row*in_dim;
    for(int v=tid; v<128; v+=WG){
      float s2=0.0f;
      for(int d=v; d<in_dim; d+=128){ const float t=x[rb2+d]; s2+=t*t; }
      rs2[v]=s2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int r2=64;r2>0;r2>>=1){
      if(tid<r2) rs2[tid]+=rs2[tid+r2];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  const float inv2=rsqrt(rs2[0]/(float)in_dim + rms2_eps);
  // The block output itself, written once. Expression order matches rms_norm_add_wg_f32's store.
  if(wg==0){
    const size_t rb2=(size_t)row*in_dim;
    for(int d=tid; d<in_dim; d+=WG)
      out_resid[rb2+d] = x[rb2+d]*inv2*vload_half(d,rms2_scale) + resid[rb2+d];
  }
#endif
// The input load, once, for every specialisation below. With RESADD the value that enters the
// GEMV is the block output the previous rms_add would have written, formed in registers.
#if RESADD
#define XL4(j) (vload4((j),x+xb)*inv2*vload_half4((j),rms2_scale) + vload4((j),resid+xb))
#define XL8(j) (vload8((j),x+xb)*inv2*vload_half8((j),rms2_scale) + vload8((j),resid+xb))
#else
#define XL4(j) vload4((j),x+xb)
#define XL8(j) vload8((j),x+xb)
#endif

#if DOT8 && INT8
  // ── pass 1: the activation scale ──────────────────────────────────────────────────────────────
  // qcom_dot8_acc needs x as unsigned int8, so we need max|x| before any product can be formed.
  // Every workgroup reads all of x anyway, so this pass is mostly cache-resident; the barrier it
  // forces is the real cost (§6.1.4).
  float amax = 0.0f;
  for(int j=tid;j<in4;j+=WG){
    float4 xv = XL4(j);
#if PRENORM
    ss += dot(xv,xv);
    xv *= vload_half4(j, rms_scale);
#endif
    amax = fmax(amax, fmax(fmax(fabs(xv.x),fabs(xv.y)), fmax(fabs(xv.z),fabs(xv.w))));
  }
  ls[tid] = amax;
#if PRENORM
  ls[tid+WG] = ss;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int r=WG/2;r>0;r>>=1){
    if(tid<r){ ls[tid]=fmax(ls[tid],ls[tid+r]);
#if PRENORM
               ls[tid+WG]+=ls[tid+r+WG];
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float amax_all = ls[0];
#if PRENORM
  const float ss_all = ls[WG];
#endif
  const float sx     = (amax_all > 0.0f) ? (amax_all * (1.0f/127.0f)) : 1.0f;
  const float inv_sx = (amax_all > 0.0f) ? (127.0f / amax_all)        : 1.0f;
  barrier(CLK_LOCAL_MEM_FENCE);     // ls is reused as an int array below

  // ── pass 2: one instruction per four products ─────────────────────────────────────────────────
  // Accumulate in int32. Worst case |Wq|<=127, xq<=255, in_dim<=4096 -> ~1.3e8, well inside int32.
#define DECL_IACC(u) int i##u = 0;
  REP(DECL_IACC)
  for(int j=tid;j<in4;j+=WG){
    float4 xv = XL4(j);
#if PRENORM
    xv *= vload_half4(j, rms_scale);
#endif
    int4 qi = convert_int4_rte(xv * inv_sx) + 128;
    qi = clamp(qi, 1, 255);
    const uint xp = (uint)qi.x | ((uint)qi.y<<8) | ((uint)qi.z<<16) | ((uint)qi.w<<24);
    // The weight uint is loaded straight from memory, so its byte order already matches the order
    // we packed x in: element 0 in the low byte. Both operands must agree, nothing more.
    // DOT8_SWAP picks which operand the instruction treats as signed. §9.5.1 says p0 is the signed
    // one and the weights go there; its own example builds p0 from `uchar p0a = -11;`, which says
    // the opposite. Rather than argue with the document, both orders get built and the device is
    // asked which one produces music.
#ifndef DOT8_SWAP
#define DOT8_SWAP 0
#endif
#if DOT8_SWAP
#define DO_DOT8(u) { const uint wp = ((const __global uint*)(W + (size_t)(n0+u)*in_dim))[j];        \
                     i##u = qcom_dot8_acc(xp, wp, i##u); }
#else
#define DO_DOT8(u) { const uint wp = ((const __global uint*)(W + (size_t)(n0+u)*in_dim))[j];        \
                     i##u = qcom_dot8_acc(wp, xp, i##u); }
#endif
    REP(DO_DOT8)
  }
  __local int* lsi = (__local int*)ls;      // integer reduction: a float would lose bits past 2^24
#define STORE_IACC(u) lsi[tid+(u)*WG] = i##u;
  REP(STORE_IACC)
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int r=WG/2;r>0;r>>=1){
    if(tid<r){
#define RED_IACC(u) lsi[tid+(u)*WG] += lsi[tid+r+(u)*WG];
      REP(RED_IACC)
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#if PRENORM
  const float inv = rsqrt(ss_all/(float)in_dim + rms_eps);
#endif
  if(tid<NOUT && n0+tid<out_dim){
    // sw*sx*( sum Wq*xq - 128*sum Wq ) — the second term is the zero-point correction, precomputed
    // per row at quantization time.
    float v = sx * wscale[n0+tid] * (float)(lsi[tid*WG] - 128 * wrowsum[n0+tid]);
#if PRENORM
    v *= inv;
#endif
#if BIAS
    v += vload_half(n0+tid,bias);
#endif
#if GELU
    v = 0.5f*v*(1.0f + erf(v*0.70710678118654752440f));
#endif
    out[(size_t)row*out_dim+n0+tid]=v;
  }
  return;
#endif  // DOT8 && INT8

  // ── wide pass: 128-bit weight loads (int8 only) ───────────────────────────────────────────────
  // Gated on INT8 as well as WIDE. The fp16 wide path (vload_half8) measured SLOWER on the 620
  // (gemv_fused 407 -> 7531 us/call before the named-accumulator fix, still slower after), and fp16
  // already moves 64 bits per work item — it has half as far to go as int8's 32.
#if IMG && !WIDE
#error "IMG requires WIDE — the image path exists only in the 128-bit step"
#endif
#if WIDE && INT8
  const int nvec = in_dim / WSTEP;
  for(int j=tid;j<nvec;j+=WG){
#if INT8
    // x as four NAMED float4s, not one float16 — see the accumulator note. Sixteen x values per
    // step, matching the sixteen weights the packed load below brings in.
    float4 x0 = XL4(j*4+0), x1 = XL4(j*4+1);
    float4 x2 = XL4(j*4+2), x3 = XL4(j*4+3);
#if PRENORM
    ss += dot(x0,x0) + dot(x1,x1) + dot(x2,x2) + dot(x3,x3);
    x0 *= vload_half4(j*4+0, rms_scale); x1 *= vload_half4(j*4+1, rms_scale);
    x2 *= vload_half4(j*4+2, rms_scale); x3 *= vload_half4(j*4+3, rms_scale);
#endif
    // char16 = one 128-bit transaction, sixteen weights. The convert_float4s are transient on
    // purpose — materialising a whole float16 of weights alongside x costs 16 more live registers
    // per output and drops occupancy.
// 128-bit packed int8 load, the form §7.2.2 of the Adreno guide actually prescribes:
//
//   "multiple 8-bit data can be manually packed into one element (e.g., 64-bit/128-bit), which is
//    loaded using vloadn, and then unpacked using as_typeN function (e.g., as_char16)"
//
// The narrow path loads vload4 on a char* — THIRTY-TWO bits per work item, a quarter of the 128-bit
// transaction §6.3 says the hardware can do. Loading a uint4 (16 bytes, one transaction) and
// unpacking with as_char16 gets sixteen weights per load instead of four.
//
// An earlier attempt used vload16 directly on the char*, which is NOT the same thing — it left the
// compiler to decide the access width and it measured slower. Casting to uint4 states the width.
// Rows start at (n0+u)*in_dim with in_dim a multiple of 16 here, so the address is 16-byte aligned.
#if IMG
// One texel = 128 bits = the same sixteen weights, fetched through the texture engine and its L1.
#define WIDE_I8(u,av) {                                                                            \
      const char16 wc = as_char16(read_imagei(Wimg, kWSmp, (int2)(j, n0+u)));                      \
      const float s = dot(x0,convert_float4(wc.lo.lo)) + dot(x1,convert_float4(wc.lo.hi))          \
                    + dot(x2,convert_float4(wc.hi.lo)) + dot(x3,convert_float4(wc.hi.hi));         \
      av += (n0+u<out_dim)?s:0.0f; }
#else
#define WIDE_I8(u,av) {                                                                            \
      const uint4 pk = vload4(j, (const __global uint*)(W + (size_t)(n0+u)*in_dim));               \
      const char16 wc = as_char16(pk);                                                             \
      const float s = dot(x0,convert_float4(wc.lo.lo)) + dot(x1,convert_float4(wc.lo.hi))          \
                    + dot(x2,convert_float4(wc.hi.lo)) + dot(x3,convert_float4(wc.hi.hi));         \
      av += (n0+u<out_dim)?s:0.0f; }
#endif
#define DO_WIDE_I8(u) WIDE_I8(u, a##u)
    REP(DO_WIDE_I8)
#else
    float8 xv = XL8(j);
#if PRENORM
    ss += dot(xv.lo,xv.lo) + dot(xv.hi,xv.hi);
    xv *= vload_half8(j, rms_scale);
#endif
#define WIDE_F16(u,av) { const float8 wv = vload_half8(j, W+(size_t)(n0+u)*in_dim);            \
      const float s = dot(xv.lo,wv.lo) + dot(xv.hi,wv.hi);                                         \
      av += (n0+u<out_dim)?s:0.0f; }
#define DO_WIDE_F16(u) WIDE_F16(u, a##u)
    REP(DO_WIDE_F16)
#endif
  }
  const int tail0 = nvec*NV;   // first float4 chunk the wide pass did not cover
#else
  const int tail0 = 0;
#endif

  // ── tail: the original 4-wide step, for any in_dim the wide step does not divide ─────────────
  for(int j=tail0+tid;j<in4;j+=WG){
    float4 xv=XL4(j);
#if PRENORM
    // sum of squares uses the RAW x, exactly as rms_norm_wg_f32 does; the scale is applied to the
    // value that enters the dot product, never to the value that is squared.
    ss += dot(xv,xv);
    xv *= vload_half4(j,rms_scale);
#endif
#if INT8
#if IMG
    // With IMG there is no W pointer at all. The wide step covers every element — an image is only
    // created when in_dim is a multiple of 16 — so the tail is compiled out to nothing.
#define NARROW(u,av) { }
#else
    // Rows start on a multiple of in_dim and every in_dim here is a multiple of 4, so the char4
    // load is 4-byte aligned. Adjacent lanes read adjacent char4 — half the bytes of the fp16
    // path for the same access pattern.
#define NARROW(u,av) { const float4 wv = convert_float4(vload4(j, W+(size_t)(n0+u)*in_dim));        \
      av += (n0+u<out_dim)?dot(xv,wv):0.0f; }
#endif
#else
#define NARROW(u,av) { const float4 wv = vload_half4(j, W+(size_t)(n0+u)*in_dim);                   \
      av += (n0+u<out_dim)?dot(xv,wv):0.0f; }
#endif
#define DO_NARROW(u) NARROW(u, a##u)
    REP(DO_NARROW)
  }
#define STORE_ACC(u) ls[tid+(u)*WG] = a##u;
  REP(STORE_ACC)
#if PRENORM
  ls[tid+NOUT*WG]=ss;
#endif
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int s=WG/2;s>0;s>>=1){
    if(tid<s){
      #pragma unroll
      for(int u=0;u<NOUT;u++) ls[tid+u*WG]+=ls[tid+s+u*WG];
#if PRENORM
      ls[tid+NOUT*WG]+=ls[tid+s+NOUT*WG];
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#if PRENORM
  const float inv = rsqrt(ls[NOUT*WG]/(float)in_dim + rms_eps);
#endif
  if(tid<NOUT && n0+tid<out_dim){
    float v = ls[tid*WG];
#if INT8
    v *= wscale[n0+tid];       // dequantize once, after the whole dot product
#endif
#if PRENORM
    v *= inv;
#endif
#if BIAS
    v += vload_half(n0+tid,bias);
#endif
#if GELU
    v = 0.5f*v*(1.0f + erf(v*0.70710678118654752440f));   // same expression as gelu_f32.cl
#endif
    out[(size_t)row*out_dim+n0+tid]=v;
  }
}
