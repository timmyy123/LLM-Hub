// Append this step's k and v into the KV cache at the CURRENT position.
//
// This replaces two clEnqueueCopyBuffer calls. The copies were correct, but
// cl_qcom_recordable_queues can only capture clEnqueueNDRangeKernel (guide 80-NB295-11 Rev C
// 9.1.3) — a copy in the middle of the AR's per-frame sequence makes the whole frame unrecordable.
// Expressing the append as a kernel makes the sequence capturable, and taking the position from a
// BUFFER rather than a scalar argument is what lets one recording serve every frame: the capture
// pins the buffer handle and the value behind it is rewritten between replays. (Scalar arg override
// via cl_array_arg_qcom returns CL_INVALID_OPERATION on this driver.)
__kernel void kv_append_f32(__global const float* qkv,    // [3*HD] — q,k,v for this step
                            __global float* kcache,       // [maxpos*HD]
                            __global float* vcache,       // [maxpos*HD]
                            __global const int* posb,     // [1] current position
                            const int HD){
  const int i = get_global_id(0);
  if (i >= HD) return;
  const size_t dst = (size_t)posb[0] * HD + i;
  kcache[dst] = qkv[HD + i];        // k starts one HD block in
  vcache[dst] = qkv[2 * HD + i];    // v starts two HD blocks in
}
