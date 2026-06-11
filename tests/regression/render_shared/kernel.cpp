// render_shared kernel — the GPU half of CPU+GPU split-frame co-rendering.
//
// The kernel renders the TOP band (rows [0, split)) of a shared framebuffer; the
// host renders the BOTTOM band ([split, h)) concurrently. Both write disjoint
// pixels of the SAME single-backing-store SVM buffer, so the finished frame is
// assembled with zero copies. A grid-stride loop covers the band for any launch
// geometry. The shader is shared verbatim with the host (common.h).

#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = gridDim.x * blockDim.x;

  uint32_t* fb = reinterpret_cast<uint32_t*>((uintptr_t)arg->fb_va);
  uint32_t  w  = arg->w;
  uint32_t  count = arg->split * w;        // pixels in the GPU (top) band

  for (uint32_t idx = tid; idx < count; idx += stride) {
    uint32_t x = idx % w;
    uint32_t y = idx / w;                   // y < split, so idx == y*w + x
    fb[idx] = shade(x, y, w, arg->h);
  }
}
