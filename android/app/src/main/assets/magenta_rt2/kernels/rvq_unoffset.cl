// Convert generated global token ids (grid) → local RVQ codes [0,1023] on the GPU, so the codec can
// start on a 2nd queue without a host round-trip. Matches the host math:
//   q = i % 12 ;  codes[i] = ((grid[off+i] - 6 - q*1024) % 1024 + 1024) % 1024
// (6 = reserved/BOS ids; q*1024 packs the 12 codebooks into one shared vocabulary.)
__kernel void rvq_unoffset(__global const int* grid, __global int* codes, const int n, const int off){
  const int i=get_global_id(0); if(i>=n) return;
  const int q=i%12;
  int v=grid[off+i]-6-q*1024;
  v=((v%1024)+1024)%1024;
  codes[i]=v;
}
