// Offset buffer copy as an NDRange dispatch.
//
// Exists because a cl_qcom_recordable_queues recording can only capture
// clEnqueueNDRangeKernel — a clEnqueueCopyBuffer inside the captured frame is simply
// not recorded, so every replay silently runs on whatever the captured frame left
// behind. That is not a crash; it is a plausible-looking wrong token, which is the
// hardest kind of failure to attribute. Every copy on the AR's frame path goes
// through this instead of clEnqueueCopyBuffer.
//
// Offsets are in ELEMENTS, not bytes.
__kernel void copy_f32(__global const float* src, __global float* dst,
                       const int src_off, const int dst_off, const int n) {
    const int i = get_global_id(0);
    if (i < n) dst[dst_off + i] = src[src_off + i];
}
