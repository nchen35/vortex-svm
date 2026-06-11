#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Number of float elements in the shared buffer (the OpenCL example uses 6).
#define N_ELEMS 6

typedef struct {
  uint64_t buf_va;   // device VA of the shared float buffer
  uint32_t n;        // element count
  uint32_t _pad;
} kernel_arg_t;

#endif
