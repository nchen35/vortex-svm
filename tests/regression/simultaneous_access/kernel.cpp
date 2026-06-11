// Simultaneous-access kernel — the device side of the OpenCL fine-grained SVM
// example. The device writes the ODD-indexed elements of the shared buffer
// (p[1]=1, p[3]=3, p[5]=5) while the host concurrently writes the EVEN ones.
// Because the buffer is a single shared backing store, both sets of writes land
// in the same memory; after the kernel completes the buffer holds {0,1,2,3,4,5}.
//
// Only thread 0 does the work (the buffer is tiny); the point is the shared
// store, not parallelism.

#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid != 0)
    return;

  float* p = reinterpret_cast<float*>((uintptr_t)arg->buf_va);
  for (uint32_t i = 1; i < arg->n; i += 2) {
    p[i] = (float)i;   // device owns odd indices
  }
}
