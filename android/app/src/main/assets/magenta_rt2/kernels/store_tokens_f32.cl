// Write this frame's NCB codebook tokens into the full grid at the CURRENT frame position.
//
// Replaces a clEnqueueCopyBuffer whose destination offset was the frame index. Both properties made
// the AR frame uncapturable by cl_qcom_recordable_queues: a copy is not an NDRange dispatch, and a
// host-computed offset would be frozen at capture time. Taking the position from the same device
// buffer the attention kernels read makes one recorded frame valid for every frame.
// `base` is the ABSOLUTE frame index this render starts at. posb holds the absolute position (it is
// the same buffer the attention kernels read, which is what makes one recorded frame replayable for
// every frame), but the grid only covers THIS render — so the row is posb[0]-base, not posb[0].
//
// Without the subtraction a continued streaming session writes past the end of the grid: request
// two runs at absolute frames 100..199 against a grid sized for 100, every token lands out of
// bounds, and the render returns whatever the grid was allocated with. base is constant for the
// whole render, so a recording stays valid.
__kernel void store_tokens_f32(__global const int* tok,    // [NCB] this frame's tokens
                               __global int* grid,          // [n_frames*NCB]
                               __global const int* posb,    // [1] ABSOLUTE frame index
                               const int NCB,
                               const int base){             // absolute index of this render's frame 0
  const int i = get_global_id(0);
  if (i >= NCB) return;
  const int row = posb[0] - base;
  if (row < 0) return;
  grid[(size_t)row * NCB + i] = tok[i];
}
